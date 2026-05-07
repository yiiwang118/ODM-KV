"""Score → bit-level mapping."""
from __future__ import annotations

from typing import Optional

import torch

DEFAULT_BIT_LEVELS: tuple[int, ...] = (0, 1, 2, 4, 8)


def scores_to_bits(
    scores: torch.Tensor,
    bit_levels: tuple[int, ...] = DEFAULT_BIT_LEVELS,
) -> torch.Tensor:
    """Map scores in [0, 1] to bit levels via uniform partitioning.

    Splits [0, 1] into ``len(bit_levels)`` equal intervals and assigns each
    score to the corresponding bit level (sorted ascending).

    Parameters
    ----------
    scores:
        Arbitrary-shape float tensor with values in [0, 1].
    bit_levels:
        Sorted tuple of bit levels to assign, e.g. (0, 1, 2, 4, 8).
        0 means eviction; 16 means keep fp16 (no quantisation).

    Returns
    -------
    torch.Tensor
        Integer tensor of same shape as ``scores``.
    """
    assert len(bit_levels) >= 1
    bits_t = torch.tensor(sorted(bit_levels), dtype=torch.int32, device=scores.device)

    if len(bit_levels) == 1:
        return bits_t[0].expand_as(scores).clone()

    # n intervals → n-1 interior split points at 1/n, 2/n, ..., (n-1)/n
    n = len(bit_levels)
    thresholds = torch.linspace(0.0, 1.0, n + 1)[1:-1].to(scores.device)
    bucket = torch.bucketize(scores.float(), thresholds)   # values in [0, n-1]
    return bits_t[bucket]


def ratio_scores_to_bits(
    scores: torch.Tensor,
    bit_levels: tuple[int, ...],
    ratios: tuple[float, ...],
    sink_mask: Optional[torch.Tensor] = None,
    sink_bits: int = 16,
    per_head: bool = True,
) -> torch.Tensor:
    """Map scores to bit levels with user-specified proportions.

    Rank-based N-tier allocation: sort non-sink tokens by importance score
    (ascending), then assign the bottom ``ratios[0]`` fraction to
    ``bit_levels[0]``, the next ``ratios[1]`` fraction to ``bit_levels[1]``,
    and so on.  ``bit_levels`` and ``ratios`` must have the same length and
    ``ratios`` must sum to 1 (tolerance 1e-3, auto-normalised otherwise).

    The implied average bit budget is ``sum(r_i * b_i)``.

    Fully vectorised — runs in O(N log N) on the GPU via a single argsort.

    Parameters
    ----------
    scores:
        Float tensor of shape ``[*shape]``.  Only the **ranking** matters.
    bit_levels:
        Candidate bit widths in ascending order, e.g. ``(1, 2, 4, 8, 16)``.
        0 = eviction; 16 = keep fp16 (no quantisation).
    ratios:
        Fraction of non-sink tokens for each level, same length as
        ``bit_levels``.  Must sum to ~1.0.
    sink_mask:
        Bool tensor, same shape as ``scores``.  ``True`` → ``sink_bits``.
    sink_bits:
        Bit width for sink tokens (default ``16`` = fp16).
    """
    assert len(bit_levels) == len(ratios), (
        f"bit_levels ({len(bit_levels)}) and ratios ({len(ratios)}) must have the same length"
    )
    shape = scores.shape
    device = scores.device

    # per_head=True: when scores has shape [..., H, T], allocate per (last-dim
    # group). Matches kvpress ScorerPress.compress which does
    # scores.topk(n_kept, dim=-1).indices — per-head top-k. Without this,
    # cross-head sorting unfairly evicts more from heads with flatter
    # distributions.
    if per_head and scores.dim() >= 2:
        out = torch.empty_like(scores, dtype=torch.int32)
        # Iterate over leading dims (everything except last)
        scores_view = scores.reshape(-1, shape[-1])
        out_view = out.reshape(-1, shape[-1])
        if sink_mask is not None:
            sm_view = sink_mask.reshape(-1, shape[-1])
        else:
            sm_view = None
        for i in range(scores_view.shape[0]):
            sub_sm = sm_view[i] if sm_view is not None else None
            out_view[i] = ratio_scores_to_bits(
                scores_view[i], bit_levels, ratios, sub_sm, sink_bits,
                per_head=False,
            )
        return out

    flat = scores.flatten()
    n = flat.shape[0]
    result = torch.empty(n, dtype=torch.int32, device=device)

    # ── Fix sink tokens ──────────────────────────────────────────────────
    if sink_mask is not None:
        flat_sink = sink_mask.flatten().bool()
        result[flat_sink] = sink_bits
        non_sink = ~flat_sink
    else:
        non_sink = torch.ones(n, dtype=torch.bool, device=device)

    non_sink_idx = non_sink.nonzero(as_tuple=True)[0]
    n_active = non_sink_idx.shape[0]
    if n_active == 0:
        return result.reshape(shape)

    # Normalise ratios
    ratios_t = torch.tensor(ratios, dtype=torch.float64)
    ratios_t = ratios_t / ratios_t.sum()

    # Sort ascending by importance
    order = flat[non_sink_idx].argsort()

    # Assign bit levels by cumulative ratio boundaries
    bits = torch.empty(n_active, dtype=torch.int32, device=device)
    levels = sorted(zip(bit_levels, ratios_t.tolist()), key=lambda x: x[0])
    cursor = 0
    for i, (b, r) in enumerate(levels):
        if i == len(levels) - 1:
            # Last tier gets all remaining to avoid rounding gaps
            count = n_active - cursor
        else:
            count = int(round(r * n_active))
            count = min(count, n_active - cursor)
        bits[order[cursor : cursor + count]] = b
        cursor += count

    result[non_sink_idx] = bits
    return result.reshape(shape)


def budget_scores_to_bits(
    scores: torch.Tensor,
    bit_levels: tuple[int, ...],
    target_avg_bits: float,
    sink_mask: Optional[torch.Tensor] = None,
    sink_bits: int = 16,
    protect_frac: float = 0.1,
) -> torch.Tensor:
    """Map scores to bit levels under an average-bit budget.

    Greedy rank-based allocation: sort non-sink tokens by importance score
    (descending), start everyone at the lowest level, then sweep through
    level transitions upgrading the highest-score tokens first until the
    budget is exhausted.

    Fully vectorised — runs in O(K · N) on the GPU (plus one O(N log N)
    argsort), where K = len(bit_levels).

    Parameters
    ----------
    scores:
        Float tensor of shape ``[*shape]``.  Only the **ranking** matters;
        monotonic transforms produce identical output.
    bit_levels:
        Candidate bit widths, e.g. ``(0, 4, 8)``.
    target_avg_bits:
        Desired average bits for **non-sink** tokens.
    sink_mask:
        Bool tensor, same shape as ``scores``.  ``True`` → ``sink_bits``.
    sink_bits:
        Bit width for sink tokens (default ``16`` = fp16).
    protect_frac:
        Fraction of non-sink tokens to protect at the highest level.
    """
    levels = sorted(set(bit_levels))
    shape = scores.shape
    device = scores.device
    flat = scores.flatten()
    n = flat.shape[0]
    result = torch.empty(n, dtype=torch.int32, device=device)

    # ── Fix sink tokens ──────────────────────────────────────────────────
    if sink_mask is not None:
        flat_sink = sink_mask.flatten().bool()
        result[flat_sink] = sink_bits
        non_sink = ~flat_sink
    else:
        non_sink = torch.ones(n, dtype=torch.bool, device=device)

    non_sink_idx = non_sink.nonzero(as_tuple=True)[0]
    n_active = non_sink_idx.shape[0]
    if n_active == 0:
        return result.reshape(shape)

    adjusted = max(float(levels[0]), min(float(levels[-1]), target_avg_bits))

    # ── Trivial cases ────────────────────────────────────────────────────
    if len(levels) == 1:
        result[non_sink_idx] = levels[0]
        return result.reshape(shape)
    if adjusted <= float(levels[0]):
        result[non_sink_idx] = levels[0]
        return result.reshape(shape)
    if adjusted >= float(levels[-1]):
        result[non_sink_idx] = levels[-1]
        return result.reshape(shape)

    non_sink_scores = flat[non_sink_idx]

    # Sort by score descending — highest-priority tokens first
    order = non_sink_scores.argsort(descending=True)

    # Start everyone at the lowest level
    bits = torch.full((n_active,), levels[0], dtype=torch.int32, device=device)
    remaining = adjusted * n_active - float(levels[0]) * n_active

    # Sweep through level transitions, upgrading highest-score tokens first
    for i in range(len(levels) - 1):
        lo, hi = levels[i], levels[i + 1]
        delta = float(hi - lo)
        if delta <= 0 or remaining < delta - 1e-9:
            continue
        # Tokens currently at level lo, in priority order
        at_lo = bits[order] == lo
        n_at_lo = int(at_lo.sum().item())
        n_upgrade = min(n_at_lo, int(remaining / delta + 1e-9))
        if n_upgrade <= 0:
            continue
        # Select the first n_upgrade among them (highest score)
        cum = at_lo.cumsum(dim=0)
        upgrade_mask = at_lo & (cum <= n_upgrade)
        bits[order[upgrade_mask]] = hi
        remaining -= n_upgrade * delta

    _ = protect_frac  # kept for API compatibility
    result[non_sink_idx] = bits
    return result.reshape(shape)


def compute_ratios(
    target_avg_bits: float,
    bit_levels: tuple[int, ...],
    *,
    evict_frac: float = 0.0,
    protect_frac: float = 0.0,
) -> tuple[float, ...]:
    """Compute ratios for ``ratio_scores_to_bits`` that hit a target average.

    Fixes *evict_frac* at the lowest level and *protect_frac* at the highest,
    then distributes the remaining fraction between the two interior levels
    that bracket the effective middle-target to hit the budget exactly.

    Parameters
    ----------
    target_avg_bits:
        Desired average bit-width.
    bit_levels:
        Sorted candidate levels, e.g. ``(0, 2, 4, 8)``.
    evict_frac:
        Fraction pinned to the lowest level (e.g. 0-bit eviction).
    protect_frac:
        Fraction pinned to the highest level (e.g. 8-bit full precision).

    Returns
    -------
    tuple[float, ...]
        Ratios aligned with *bit_levels*, summing to 1.0, with implied
        average ≈ *target_avg_bits*.

    Examples
    --------
    >>> compute_ratios(2.0, (0, 2, 4, 8), evict_frac=0.25, protect_frac=0.05)
    (0.25, 0.6, 0.1, 0.05)       # avg = 0+1.2+0.4+0.4 = 2.0
    >>> compute_ratios(3.0, (0, 2, 4, 8), evict_frac=0.10, protect_frac=0.05)
    (0.1, 0.3571..., 0.4928..., 0.05)   # avg ≈ 3.0
    """
    levels = sorted(set(int(b) for b in bit_levels))
    K = len(levels)
    if K < 2:
        return (1.0,)

    lo, hi = float(levels[0]), float(levels[-1])
    target = max(lo, min(hi, target_avg_bits))

    evict_frac = max(0.0, min(evict_frac, 0.99))
    protect_frac = max(0.0, min(protect_frac, 0.99))
    if evict_frac + protect_frac >= 1.0:
        protect_frac = max(0.0, 0.99 - evict_frac)

    middle_frac = 1.0 - evict_frac - protect_frac
    ratios = [0.0] * K
    ratios[0] = evict_frac
    ratios[-1] = protect_frac

    if middle_frac <= 1e-9:
        return tuple(ratios)

    # Effective target for the middle portion
    middle_target = (target - evict_frac * lo - protect_frac * hi) / middle_frac
    middle_target = max(lo, min(hi, middle_target))

    # Find two levels that bracket middle_target
    bracket_lo = max(lv for lv in levels if lv <= middle_target)
    bracket_hi = min(lv for lv in levels if lv >= middle_target)

    if bracket_lo == bracket_hi:
        # middle_target exactly matches a level
        ratios[levels.index(bracket_lo)] += middle_frac
    else:
        span = float(bracket_hi - bracket_lo)
        r_hi = middle_frac * (middle_target - bracket_lo) / span
        r_lo = middle_frac - r_hi
        ratios[levels.index(bracket_lo)] += r_lo
        ratios[levels.index(bracket_hi)] += r_hi

    return tuple(ratios)


# ---------------------------------------------------------------------------
# OCS (Outlier Channel Separation) helpers
# ---------------------------------------------------------------------------

def effective_bits(b: int, n_outlier: int = 0, d: int = 128, outlier_min_bits: int = 4) -> float:
    """Actual average bits per element with OCS.

    Outlier channels use ``max(b, outlier_min_bits)`` scalar bits; regular
    channels use ``b`` TurboQuant bits.  Extra cost only when ``b < outlier_min_bits``.
    """
    if b == 0 or b >= 16 or n_outlier <= 0:
        return float(b)
    b_out = max(b, outlier_min_bits)
    return b + n_outlier * max(0, b_out - b) / d


# ---------------------------------------------------------------------------
# Epsilon calibration
# ---------------------------------------------------------------------------

_epsilon_cache: dict[tuple, dict[int, float]] = {}


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
        d, tuple(sorted(bit_levels)), seed, value_group_size,
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


# ---------------------------------------------------------------------------
# Lagrangian-optimal bit allocation
# ---------------------------------------------------------------------------

def optimal_scores_to_bits(
    scores: torch.Tensor,
    bit_levels: tuple[int, ...],
    target_avg_bits: float,
    epsilon: dict[int, float],
    sink_mask: Optional[torch.Tensor] = None,
    sink_bits: int = 16,
    eviction_cost: float = 5.0,
    n_outlier: int = 0,
    head_dim: int = 128,
    outlier_min_bits: int = 4,
    above_target_alpha: float = 1.0,
) -> torch.Tensor:
    """Lagrangian-optimal per-token bit allocation under a budget.

    For each non-sink token *i* with importance score *s_i*, assigns

        b_i* = argmin_b { s_i · ε'(b) + λ · effective(b) }

    where ``effective(b)`` accounts for OCS outlier channel overhead.

    When ``above_target_alpha < 1``, the ε benefit for levels above the
    target is compressed:

        ε'(b) = ε(target) · (ε(b) / ε(target))^α   for b > target

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
        Multiplier on ``ε(0)`` to penalise eviction.  Default 5.0.
    n_outlier : int
        Number of outlier channels per head for OCS.  0 = disabled.
    head_dim : int
        Head dimension, used for effective bits calculation.
    above_target_alpha : float
        Compression exponent for ε above target.  1.0 = no compression
        (original behaviour).  Lower values reduce the incentive to
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
        return result.reshape(shape)

    # ── Trivial cases ────────────────────────────────────────────────
    target = max(float(levels[0]), min(float(levels[-1]), target_avg_bits))
    if len(levels) == 1:
        result[non_sink_idx] = levels[0]
        return result.reshape(shape)

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
    def _assign(lam: float) -> tuple[torch.Tensor, float]:
        # cost[k, i] = s_i × ε'(b_k) + λ × eff(b_k)
        cost = s.unsqueeze(0) * eps_t.unsqueeze(1) + lam * bits_t.unsqueeze(1)
        chosen = cost.argmin(dim=0)                                # [n_active]
        avg = bits_t[chosen].mean().item()
        return chosen, avg

    # ── Binary search for λ ──────────────────────────────────────────
    # Large λ penalises bits → pushes towards 0-bit → low avg.
    # Small λ → pushes towards max-bit → high avg.
    lo, hi = 0.0, 1.0

    # Widen upper bound until avg is below target
    _, avg_at_hi = _assign(hi)
    while avg_at_hi > target and hi < 1e8:
        hi *= 10.0
        _, avg_at_hi = _assign(hi)

    for _ in range(64):
        mid = (lo + hi) * 0.5
        _, avg = _assign(mid)
        if avg > target:
            lo = mid
        else:
            hi = mid

    chosen, avg = _assign((lo + hi) * 0.5)

    # ── Post-processing: close budget gap via greedy adjustment ──────
    # Work with effective bits for budget math; chosen indices map to
    # both effective costs (bits_t) and nominal levels (nominal_t).
    eff_assigned = bits_t[chosen]                                  # [n_active] eff float
    nom_assigned = nominal_t[chosen].float()                       # [n_active] nominal
    gap = (avg - target) * n_active                                # total excess eff bits

    if abs(gap) > 0.5:
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
    return result.reshape(shape)
