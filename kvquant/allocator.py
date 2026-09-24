"""Quantizer calibration and joint eviction/precision allocation for ODM-KV."""
from __future__ import annotations
from typing import Optional
import torch

DEFAULT_BIT_LEVELS = (0, 2, 3, 4, 8, 16)
_epsilon_cache: dict[tuple, dict[int, float]] = {}

def effective_bits(b: int, n_outlier: int = 0, d: int = 128, outlier_min_bits: int = 4) -> float:
    """Actual average bits per element with OCS.

    Outlier channels use ``max(b, outlier_min_bits)`` scalar bits; regular
    channels use ``b`` TurboQuant bits.  Extra cost only when ``b < outlier_min_bits``.
    """
    if b == 0 or b >= 16 or n_outlier <= 0:
        return float(b)
    b_out = max(b, outlier_min_bits)
    return b + n_outlier * max(0, b_out - b) / d


@torch.no_grad()
def calibrate_epsilon(
    d: int,
    bit_levels: tuple[int, ...],
    *,
    n_samples: int = 4096,
    seed: int = 42,
    device: torch.device | str = "cpu",
    value_group_size: int = 32,
    n_outlier: int = 0,
    outlier_min_bits: int = 4,
    value_quantizer: str = "minmax",
) -> dict[int, float]:
    """Measure combined key + value relative quantisation MSE per bit level.

    When ``n_outlier > 0`` (OCS mode), key error is measured with the
    outlier channel separation scheme: *n_outlier* channels quantised via
    scalar MinMax at ``max(b, outlier_min_bits)`` bits, remaining channels
    via TurboQuantMSE at *b* bits.

    ``value_quantizer`` selects which quantiser is used to measure value
    reconstruction error:

    * ``"minmax"``: per-group scalar min/max at bit *b* (default for
      backwards-compat).
    * ``"mse"``: TurboQuantMSE codebook over the whole head dim at bit
      *b* (matches the adaptive backend when ``value_quantizer="mse"``).

    The result is cached globally so repeated calls are free — the cache
    key is over every parameter that affects the output so different
    quantisers don't collide.
    """
    device = torch.device(device) if isinstance(device, str) else device
    cache_key = (
        d, tuple(sorted(bit_levels)), n_samples, seed, value_group_size,
        n_outlier, outlier_min_bits, value_quantizer,
    )
    if cache_key in _epsilon_cache:
        return _epsilon_cache[cache_key]

    from kvquant.tq_backend import TurboQuantMSE, _quantize_values_minmax, _dequantize_values_minmax

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    x = torch.randn(n_samples, d, generator=gen).to(device=device, dtype=torch.float16)
    x_norm_sq = x.float().pow(2).sum(-1).mean().item()

    epsilon: dict[int, float] = {}
    for b in sorted(set(bit_levels)):
        if b == 0:
            epsilon[0] = 1.0
            continue
        if b >= 16:
            epsilon[b] = 0.0
            continue

        # ── Key error ───────────────────────────────────────────────
        if n_outlier > 0 and n_outlier < d:
            # OCS: outlier channels → scalar MinMax, regular → TurboQuant
            d_reg = d - n_outlier
            b_out = max(b, outlier_min_bits)
            # Regular channels
            x_reg = x[:, n_outlier:]
            kq = TurboQuantMSE(d_reg, b, device=device, dtype=torch.float16, seed=seed)
            k_reg_recon = kq.dequantize(kq.quantize(x_reg))
            # Outlier channels — scalar MinMax at b_out
            x_out = x[:, :n_outlier]
            gs_out = min(value_group_size, n_outlier) or n_outlier
            if n_outlier % gs_out != 0:
                gs_out = n_outlier
            vq_out = _quantize_values_minmax(x_out, b_out, gs_out)
            k_out_recon = _dequantize_values_minmax(vq_out, gs_out)
            k_recon = torch.cat([k_out_recon, k_reg_recon], dim=-1)
            eps_key = (x.float() - k_recon.float()).pow(2).sum(-1).mean().item() / max(x_norm_sq, 1e-12)
        else:
            # No OCS: full-dim TurboQuant
            kq = TurboQuantMSE(d, b, device=device, dtype=torch.float16, seed=seed)
            k_recon = kq.dequantize(kq.quantize(x))
            eps_key = (x.float() - k_recon.float()).pow(2).sum(-1).mean().item() / max(x_norm_sq, 1e-12)

        # ── Value error ──────────────────────────────────────────────
        # Match the actual value quantiser used at runtime. Mismatched
        # calibration would bias Lagrangian bit allocation by whatever
        # the MSE gap is between MinMax and TurboQuantMSE at this bit.
        # MinMax at 1-bit has n_levels=1 (useless); fall back to MSE.
        use_mse_value = (value_quantizer == "mse") or (b == 1)
        if use_mse_value:
            vq_mse = TurboQuantMSE(d, b, device=device, dtype=torch.float16, seed=seed + 2000)
            v_recon = vq_mse.dequantize(vq_mse.quantize(x))
            eps_val = (x.float() - v_recon.float()).pow(2).sum(-1).mean().item() / max(x_norm_sq, 1e-12)
        elif d % value_group_size == 0:
            vq = _quantize_values_minmax(x, b, value_group_size)
            v_recon = _dequantize_values_minmax(vq, value_group_size)
            eps_val = (x.float() - v_recon.float()).pow(2).sum(-1).mean().item() / max(x_norm_sq, 1e-12)
        else:
            eps_val = eps_key

        epsilon[b] = (eps_key + eps_val) / 2.0

    _epsilon_cache[cache_key] = epsilon
    return epsilon


def optimal_scores_to_bits(
    scores: torch.Tensor,
    bit_levels: tuple[int, ...],
    target_avg_bits: float,
    epsilon: dict[int, float],
    sink_mask: Optional[torch.Tensor] = None,
    sink_bits: int = 16,
    eviction_cost: float = 0.5,
    n_outlier: int = 0,
    head_dim: int = 128,
    outlier_min_bits: int = 4,
    above_target_alpha: float = 1.0,
    fixed_lambda: Optional[float] = None,
    return_lambda: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, float]:
    """Lagrangian per-token allocation followed by heuristic budget repair.

    The raw assignment at a fixed λ is optimal for the budget it attains.
    When binary search does not land on the requested discrete budget, the
    final score-ordered promotion/demotion pass is a feasibility heuristic;
    this function does not claim that repaired output is globally optimal.

    For each non-sink token *i* with importance score *s_i*, assigns

        b_i* = argmin_b { s_i · ε'(b) + λ · effective(b) }

    where ``effective(b)`` accounts for OCS outlier channel overhead.

    When ``above_target_alpha < 1``, the ε benefit for levels above the
    target is compressed:

        ε'(b) = (1 - α) · ε(target) + α · ε(b)   for b > target

    This prevents over-allocation to high bit levels where marginal
    quality improvement is negligible (e.g. 8-bit vs 4-bit).
    Levels at or below target are unaffected.

    Parameters
    ----------
    scores : torch.Tensor
        Float tensor of shape ``[*shape]``.  Higher = more important.
    bit_levels : tuple[int, ...]
        Candidate bit widths, e.g. ``(0, 2, 4, 8)``.
    target_avg_bits : float
        Desired average bits for non-sink tokens (in effective bits).
    epsilon : dict[int, float]
        Relative MSE per bit level, from ``calibrate_epsilon()``.
    sink_mask, sink_bits : optional
        As in ``ratio_scores_to_bits``.
    eviction_cost : float
        Multiplier on ``ε(0)`` to price eviction.  Default 0.5.
    n_outlier : int
        Number of outlier channels per head for OCS.  0 = disabled.
    head_dim : int
        Head dimension, used for effective bits calculation.
    above_target_alpha : float
        Linear interpolation weight for ε above target.  1.0 = no compression
        (original behaviour); 0.0 flattens those levels to ε(target).
        Lower values reduce the incentive to
        allocate bits above the target.  Recommended: 0.5.
    """
    levels = sorted(set(bit_levels))
    shape = scores.shape
    device = scores.device
    flat = scores.flatten().float()
    n = flat.shape[0]
    result = torch.empty(n, dtype=torch.int32, device=device)

    # ── Fix sink tokens ──────────────────────────────────────────────
    if sink_mask is not None:
        flat_sink = sink_mask.flatten().bool()
        result[flat_sink] = sink_bits
        non_sink = ~flat_sink
    else:
        non_sink = torch.ones(n, dtype=torch.bool, device=device)

    non_sink_idx = non_sink.nonzero(as_tuple=True)[0]
    n_active = non_sink_idx.shape[0]
    if n_active == 0:
        bits_out = result.reshape(shape)
        return (bits_out, 0.0) if return_lambda else bits_out

    # ── Trivial cases ────────────────────────────────────────────────
    target = max(float(levels[0]), min(float(levels[-1]), target_avg_bits))
    if len(levels) == 1:
        result[non_sink_idx] = levels[0]
        bits_out = result.reshape(shape)
        return (bits_out, 0.0) if return_lambda else bits_out

    s = flat[non_sink_idx]                                         # [n_active]

    # ── Sanitise scores ──────────────────────────────────────────────
    # Scores should be in [0, 1] after _normalize_scores, but defend
    # against inf / nan / negatives that would break the Lagrangian.
    s = s.clamp(min=0.0)
    finite = s.isfinite()
    if finite.any() and not finite.all():
        s = torch.where(finite, s, s[finite].max())               # inf → max finite
    s = torch.nan_to_num(s, nan=0.0)                              # nan → 0

    K = len(levels)
    eps_raw = [epsilon.get(b, 0.0) for b in levels]
    if eviction_cost != 1.0 and levels[0] == 0:
        eps_raw[0] *= eviction_cost

    # ── Compress ε above target ──────────────────────────────────────
    # Find ε at the nominal target (interpolate between bracketing levels)
    eps_at_target = None
    if above_target_alpha < 1.0:
        # Find the level at or just below target
        below = [b for b in levels if b <= target_avg_bits and b != 0]
        eps_at_target = epsilon.get(int(target_avg_bits), None)
        if eps_at_target is None and below:
            eps_at_target = epsilon.get(below[-1], 0.0)
        if eps_at_target is not None and eps_at_target > 1e-12:
            for k, b in enumerate(levels):
                if b > target_avg_bits:
                    # Linear interpolation toward ε(target):
                    # α=1 → original, α=0 → all equal to ε(target)
                    eps_raw[k] = (1.0 - above_target_alpha) * eps_at_target \
                               + above_target_alpha * eps_raw[k]

    eps_t = torch.tensor(eps_raw, dtype=torch.float32, device=device)  # [K]
    # Use effective bits (accounts for OCS outlier overhead) as λ cost
    eff = [effective_bits(b, n_outlier, head_dim, outlier_min_bits) for b in levels]
    bits_t = torch.tensor(eff, dtype=torch.float32, device=device)     # [K]
    # Nominal levels for final assignment (still integer bit levels)
    nominal_t = torch.tensor(levels, dtype=torch.int32, device=device) # [K]

    # ── Helper: assign bits at a given λ ─────────────────────────────
    # lam may be a python float OR a 0-d GPU tensor; both broadcast cleanly.
    def _assign(lam):
        # cost[k, i] = s_i × ε'(b_k) + λ × eff(b_k)
        cost = s.unsqueeze(0) * eps_t.unsqueeze(1) + lam * bits_t.unsqueeze(1)
        chosen = cost.argmin(dim=0)                                # [n_active]
        avg = bits_t[chosen].mean()                                # 0-d GPU tensor
        return chosen, avg

    # ── Resolve λ ────────────────────────────────────────────────────
    # Large λ penalises bits → pushes towards 0-bit → low avg.
    # Small λ → pushes towards max-bit → high avg.
    if fixed_lambda is not None:
        # Pre-calibrated λ path: skip binary search. Used for streaming
        # per-layer commit where a global λ* was captured on a prior
        # prefill. The per-layer budget drift is by design — global
        # accuracy comes from applying the *same* λ everywhere.
        lam_star_t = torch.tensor(float(fixed_lambda), dtype=torch.float32, device=device)
        chosen, avg_t = _assign(lam_star_t)
        lam_star = float(fixed_lambda)
    else:
        # Tensorised binary search: keep lo/hi/mid as 0-d GPU tensors so
        # the 64-iter loop runs without Python ↔ GPU sync per iteration.
        # Previous impl called .item() once per iter (~65 syncs per
        # allocate); now we sync exactly once at the end.
        #
        # Start hi at 1e8 instead of widening incrementally — same kernel
        # count as the old typical case (hi=1.0 sufficient + 64 search +
        # 1 final = 65 calls). 64 halvings give resolution 1e8/2^64 ≈
        # 5e-12, more than enough to bracket lam* to ε.
        lo = torch.zeros((), dtype=torch.float32, device=device)
        hi = torch.full((), 1e8, dtype=torch.float32, device=device)
        target_t = torch.tensor(target, dtype=torch.float32, device=device)

        for _ in range(64):
            mid = (lo + hi) * 0.5
            _, avg_mid = _assign(mid)
            over = avg_mid > target_t
            lo = torch.where(over, mid, lo)
            hi = torch.where(over, hi, mid)

        lam_star_t = (lo + hi) * 0.5
        chosen, avg_t = _assign(lam_star_t)
        # Defer lam_star → cpu sync until we know it's needed (return_lambda)
        lam_star = None

    avg = avg_t.item()                                             # 1 sync (for gap math)

    # ── Post-processing: close budget gap via greedy adjustment ──────
    # Work with effective bits for budget math; chosen indices map to
    # both effective costs (bits_t) and nominal levels (nominal_t).
    eff_assigned = bits_t[chosen]                                  # [n_active] eff float
    nom_assigned = nominal_t[chosen].float()                       # [n_active] nominal
    gap = (avg - target) * n_active                                # total excess eff bits

    # Skip gap-closing when a pre-calibrated λ is in use: the point of
    # fixed_lambda is to mirror the *global* argmin, so we must NOT do
    # per-call promotion/demotion based on the local score distribution.
    do_gap_closing = fixed_lambda is None
    if do_gap_closing and abs(gap) > 0.5:
        order = s.argsort()                                        # ascending
        eff_ordered = eff_assigned[order]
        nom_ordered = nom_assigned[order]

        if gap > 0:
            # Over budget → demote lowest-score tokens one level down
            for i in range(K - 1):
                eff_hi = float(eff[i + 1])
                eff_lo = float(eff[i])
                delta = eff_hi - eff_lo
                if delta <= 0:
                    continue
                at_hi = (eff_ordered == eff_hi)
                n_demote = min(int(at_hi.sum().item()), int(gap / delta + 0.5))
                if n_demote <= 0:
                    continue
                cum = at_hi.cumsum(dim=0)
                demote_mask = at_hi & (cum <= n_demote)
                eff_ordered[demote_mask] = eff_lo
                nom_ordered[demote_mask] = float(levels[i])
                gap -= n_demote * delta
                if gap <= 0.5:
                    break
        else:
            # Under budget → promote highest-score tokens one level up
            gap = -gap
            for i in range(K - 2, -1, -1):
                eff_lo = float(eff[i])
                eff_hi = float(eff[i + 1])
                delta = eff_hi - eff_lo
                if delta <= 0:
                    continue
                at_lo = (eff_ordered == eff_lo)
                n_promote = min(int(at_lo.sum().item()), int(gap / delta + 0.5))
                if n_promote <= 0:
                    continue
                cum_rev = at_lo.flip(0).cumsum(dim=0).flip(0)
                promote_mask = at_lo & (cum_rev <= n_promote)
                eff_ordered[promote_mask] = eff_hi
                nom_ordered[promote_mask] = float(levels[i + 1])
                gap -= n_promote * delta
                if gap <= 0.5:
                    break

        nom_assigned = torch.empty_like(nom_ordered)
        nom_assigned[order] = nom_ordered

    result[non_sink_idx] = nom_assigned.to(torch.int32)
    bits_out = result.reshape(shape)
    if return_lambda:
        if lam_star is None:
            lam_star = lam_star_t.item()
        return bits_out, lam_star
    return bits_out
