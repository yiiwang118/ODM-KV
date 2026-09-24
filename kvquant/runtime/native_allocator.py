"""Batched, device-resident layerwise bit allocation for Realsys2.

This module freezes the allocation semantics used by
``kvquant.allocator.optimal_scores_to_bits`` while removing the two properties
that make that reference unsuitable for a serving hot path:

* requests are allocated one at a time in Python;
* allocation decisions are copied back to the host during the lambda search
  and budget-repair pass.

The implementation below is deliberately a correctness backend.  It keeps one
lambda per request, but evaluates all requests in a single tensor program.  Its
public boundary, :class:`BatchedNativeAllocator`, is intentionally independent
of the search implementation so a Triton/CUDA breakpoint selector can replace
``_search_lambda`` without changing callers or the semantic tests.

Important policy details retained here:

* one budget is solved independently for every request and layer, across all
  KV heads and tokens in that request;
* sink cells are excluded from the budget and forced to ``sink_bits``;
* 0-bit error is multiplied by ``eviction_cost`` (0.5 in production);
* the reference level-ordered score repair is applied after the lambda solve;
* the recent tail is forced exact *after* allocation, so it does not alter the
  lambda target.  This matches the current Realsys2/simulator policy even though
  the resulting physical average, including the protected tail, is above the
  nominal target.

The reference repair uses an unstable sort, so which *identity* among exactly
equal-score cells is promoted is not a specified or portable property.  This
backend makes that corner deterministic: ascending repair ties follow flattened
``(head, token)`` order and descending repair ties use the reverse order.  Bit
histograms, budget, and objective value remain identical to the reference; only
interchangeable equal-score token identities may differ.

There are no host reads of device values in the production path.  CUDA rows up
to 32K cells use vector-resident persistent Triton kernels; wider rows use
fixed-tile streaming persistent kernels.  CPU and OCS retain the exact
vectorized fallback.  No backend loops over requests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover - CPU-only developer machines
    _HAS_TRITON = False


# One persistent CTA owns one flattened request row.  RTX 4090 profiling keeps
# the vector-resident implementation through 8K cells; larger rows switch to
# the fixed-tile streaming implementation.  At 16K+ the streaming CTA is both
# faster and dramatically cheaper to compile than expanding the full row.
_MAX_PERSISTENT_CELLS = 8192

# Safety margin on the tightened lambda upper bound (see ``_initial_hi``).  The
# budget-hitting lambda* is provably <= max_active_score * max_pairwise_hull
# slope, so 2x guards fp rounding while keeping the fast-path bracket tight
# enough that ~15 bisections reach the same tags as the 64-step [0, 1e8] search.
_HI_SAFETY = 2.0


if _HAS_TRITON:

    @triton.jit
    def _persistent_lambda_search_kernel(
        SCORES,
        ACTIVE_COUNT,
        EPSILON,
        EFFECTIVE_BITS,
        LAMBDA,
        HI_INIT,
        N: tl.constexpr,
        NUM_LEVELS: tl.constexpr,
        TARGET: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Solve one request's lambda entirely inside one persistent CTA.

        Eager PyTorch's segment-reduce launches a large reduction kernel once
        per bisection step.  Here every request owns one CTA; its score row is
        retained as the persistent working set while all search iterations run
        device-side.  The no-OCS fast path has integral effective costs, so the
        reduction is exact in fp32 for all supported row widths.
        """
        request = tl.program_id(0)
        offset = tl.arange(0, BLOCK_N)
        count = tl.load(ACTIVE_COUNT + request)
        valid = offset < count
        in_bounds = offset < N
        score = tl.load(
            SCORES + request * N + offset,
            mask=in_bounds,
            other=0.0,
        ).to(tl.float32)

        lo = 0.0
        hi = tl.load(HI_INIT + request)
        # ``tl.range`` emits a device loop instead of unrolling 64 copies of
        # the large candidate/reduction body into the compiled kernel.
        for _ in tl.range(0, SEARCH_STEPS):
            mid = (lo + hi) * 0.5
            eps = tl.load(EPSILON)
            width = tl.load(EFFECTIVE_BITS)
            best_cost = score * eps + mid * width
            best_width = tl.full([BLOCK_N], width, tl.float32)
            for level in tl.static_range(1, NUM_LEVELS):
                eps = tl.load(EPSILON + level)
                width = tl.load(EFFECTIVE_BITS + level)
                cost = score * eps + mid * width
                take = cost < best_cost
                best_cost = tl.where(take, cost, best_cost)
                best_width = tl.where(take, width, best_width)

            total = tl.sum(tl.where(valid, best_width, 0.0), axis=0)
            denominator = count.to(tl.float32)
            average = tl.where(count > 0, total * (1.0 / denominator), 0.0)
            over = average > TARGET
            lo = tl.where(over, mid, lo)
            hi = tl.where(over, hi, mid)

        tl.store(LAMBDA + request, (lo + hi) * 0.5)


    @triton.jit
    def _streaming_lambda_search_kernel(
        SCORES,
        ACTIVE_COUNT,
        EPSILON,
        EFFECTIVE_BITS,
        LAMBDA,
        HI_INIT,
        N,
        NUM_LEVELS: tl.constexpr,
        TARGET: tl.constexpr,
        SEARCH_STEPS: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Persistent long-row search with a fixed-size streaming tile.

        Unlike the vector-resident kernel, compile/register cost is independent
        of context length.  Integral no-OCS widths make the tile sums and their
        scalar accumulation exact in fp32 up to serving-scale row lengths.
        """
        request = tl.program_id(0)
        lane = tl.arange(0, BLOCK_N)
        count = tl.load(ACTIVE_COUNT + request)
        lo = 0.0
        hi = tl.load(HI_INIT + request)

        for _ in tl.range(0, SEARCH_STEPS):
            mid = (lo + hi) * 0.5
            total = 0.0
            for start in tl.range(0, N, BLOCK_N):
                offset = start + lane
                in_bounds = offset < N
                active = offset < count
                score = tl.load(
                    SCORES + request * N + offset,
                    mask=in_bounds,
                    other=0.0,
                ).to(tl.float32)
                eps = tl.load(EPSILON)
                width = tl.load(EFFECTIVE_BITS)
                best_cost = score * eps + mid * width
                best_width = tl.full([BLOCK_N], width, tl.float32)
                for level in tl.static_range(1, NUM_LEVELS):
                    eps = tl.load(EPSILON + level)
                    width = tl.load(EFFECTIVE_BITS + level)
                    cost = score * eps + mid * width
                    take = cost < best_cost
                    best_cost = tl.where(take, cost, best_cost)
                    best_width = tl.where(take, width, best_width)
                total += tl.sum(
                    tl.where(active, best_width, 0.0), axis=0,
                )

            denominator = count.to(tl.float32)
            average = tl.where(count > 0, total * (1.0 / denominator), 0.0)
            over = average > TARGET
            lo = tl.where(over, mid, lo)
            hi = tl.where(over, hi, mid)

        tl.store(LAMBDA + request, (lo + hi) * 0.5)


    @triton.jit
    def _persistent_budget_repair_kernel(
        SCORES,
        ACTIVE_COUNT,
        CHOSEN,
        AVERAGE,
        EFFECTIVE_BITS,
        NOMINAL_BITS,
        OUT,
        N: tl.constexpr,
        NUM_LEVELS: tl.constexpr,
        TARGET: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Stable level repair using grouped extrema, without a full sort."""
        request = tl.program_id(0)
        offset = tl.arange(0, BLOCK_N)
        count = tl.load(ACTIVE_COUNT + request)
        in_bounds = offset < N
        active = offset < count
        score = tl.load(
            SCORES + request * N + offset,
            mask=in_bounds,
            other=0.0,
        ).to(tl.float32)
        current = tl.load(
            CHOSEN + request * N + offset,
            mask=in_bounds,
            other=0,
        ).to(tl.int32)
        average = tl.load(AVERAGE + request).to(tl.float64)
        gap = (average - TARGET) * count.to(tl.float64)

        over_gap = gap
        for level in tl.static_range(0, NUM_LEVELS - 1):
            low_width = tl.load(EFFECTIVE_BITS + level).to(tl.float64)
            high_width = tl.load(EFFECTIVE_BITS + level + 1).to(tl.float64)
            delta = high_width - low_width
            candidate = active & (current == level + 1)
            available = tl.sum(candidate.to(tl.int32), axis=0)
            requested = (over_gap / delta + 0.5).to(tl.int32)
            requested = tl.where(over_gap > 0.5, tl.maximum(requested, 0), 0)
            take_count = tl.minimum(available, requested)
            remaining = take_count
            # At a normal (non-tied) breakpoint take_count is at most a few
            # cells.  Exact-score ties are consumed as one group by cumsum, so
            # the adversarial all-equal policy is still one iteration.
            while remaining > 0:
                threshold = tl.min(
                    tl.where(candidate, score, float("inf")), axis=0,
                )
                equal = candidate & (score == threshold)
                equal_count = tl.sum(equal.to(tl.int32), axis=0)
                rank = tl.cumsum(equal.to(tl.int32), axis=0)
                selected = equal & (rank <= remaining)
                current = tl.where(selected, level, current)
                consumed = tl.minimum(equal_count, remaining)
                remaining -= consumed
                candidate &= ~equal
            over_gap -= take_count.to(tl.float64) * delta

        under_gap = -gap
        for level in tl.static_range(NUM_LEVELS - 2, -1, -1):
            low_width = tl.load(EFFECTIVE_BITS + level).to(tl.float64)
            high_width = tl.load(EFFECTIVE_BITS + level + 1).to(tl.float64)
            delta = high_width - low_width
            candidate = active & (current == level)
            available = tl.sum(candidate.to(tl.int32), axis=0)
            requested = (under_gap / delta + 0.5).to(tl.int32)
            requested = tl.where(under_gap > 0.5, tl.maximum(requested, 0), 0)
            take_count = tl.minimum(available, requested)
            remaining = take_count
            while remaining > 0:
                threshold = tl.max(
                    tl.where(candidate, score, -float("inf")), axis=0,
                )
                equal = candidate & (score == threshold)
                equal_count = tl.sum(equal.to(tl.int32), axis=0)
                rank = tl.cumsum(equal.to(tl.int32), axis=0, reverse=True)
                selected = equal & (rank <= remaining)
                current = tl.where(selected, level + 1, current)
                consumed = tl.minimum(equal_count, remaining)
                remaining -= consumed
                candidate &= ~equal
            under_gap -= take_count.to(tl.float64) * delta

        nominal = tl.load(NOMINAL_BITS + current)
        tl.store(OUT + request * N + offset, nominal, mask=in_bounds)


    @triton.jit
    def _streaming_budget_repair_kernel(
        SCORES,
        ACTIVE_COUNT,
        CHOSEN,
        AVERAGE,
        EFFECTIVE_BITS,
        NOMINAL_BITS,
        OUT,
        N,
        NUM_LEVELS: tl.constexpr,
        TARGET: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Stable exact repair for long rows using a fixed-size tile."""
        request = tl.program_id(0)
        lane = tl.arange(0, BLOCK_N)
        count = tl.load(ACTIVE_COUNT + request)
        row = request * N

        # OUT is a level-index workspace until the final nominal-bit pass.
        for start in tl.range(0, N, BLOCK_N):
            offset = start + lane
            in_bounds = offset < N
            current = tl.load(
                CHOSEN + row + offset, mask=in_bounds, other=0,
            ).to(tl.int32)
            tl.store(OUT + row + offset, current, mask=in_bounds)

        average = tl.load(AVERAGE + request).to(tl.float64)
        gap = (average - TARGET) * count.to(tl.float64)

        over_gap = gap
        for level in tl.static_range(0, NUM_LEVELS - 1):
            low_width = tl.load(EFFECTIVE_BITS + level).to(tl.float64)
            high_width = tl.load(EFFECTIVE_BITS + level + 1).to(tl.float64)
            delta = high_width - low_width
            available = 0
            for start in tl.range(0, N, BLOCK_N):
                offset = start + lane
                active = offset < count
                current = tl.load(
                    OUT + row + offset, mask=offset < N, other=0,
                )
                available += tl.sum(
                    (active & (current == level + 1)).to(tl.int32), axis=0,
                )
            requested = (over_gap / delta + 0.5).to(tl.int32)
            requested = tl.where(over_gap > 0.5, tl.maximum(requested, 0), 0)
            take_count = tl.minimum(available, requested)
            remaining = take_count
            while remaining > 0:
                threshold = float("inf")
                for start in tl.range(0, N, BLOCK_N):
                    offset = start + lane
                    in_bounds = offset < N
                    active = offset < count
                    current = tl.load(
                        OUT + row + offset, mask=in_bounds, other=0,
                    )
                    score = tl.load(
                        SCORES + row + offset, mask=in_bounds, other=0.0,
                    ).to(tl.float32)
                    candidate = active & (current == level + 1)
                    tile_min = tl.min(
                        tl.where(candidate, score, float("inf")), axis=0,
                    )
                    threshold = tl.minimum(threshold, tile_min)

                group_count = 0
                for start in tl.range(0, N, BLOCK_N):
                    offset = start + lane
                    in_bounds = offset < N
                    active = offset < count
                    current = tl.load(
                        OUT + row + offset, mask=in_bounds, other=0,
                    )
                    score = tl.load(
                        SCORES + row + offset, mask=in_bounds, other=0.0,
                    ).to(tl.float32)
                    equal = active & (current == level + 1) & (score == threshold)
                    local_rank = tl.cumsum(equal.to(tl.int32), axis=0)
                    selected = equal & (local_rank + group_count <= remaining)
                    tl.store(
                        OUT + row + offset,
                        tl.where(selected, level, current),
                        mask=in_bounds,
                    )
                    group_count += tl.sum(equal.to(tl.int32), axis=0)
                consumed = tl.minimum(group_count, remaining)
                remaining -= consumed
            over_gap -= take_count.to(tl.float64) * delta

        under_gap = -gap
        chunks = (N + BLOCK_N - 1) // BLOCK_N
        for level in tl.static_range(NUM_LEVELS - 2, -1, -1):
            low_width = tl.load(EFFECTIVE_BITS + level).to(tl.float64)
            high_width = tl.load(EFFECTIVE_BITS + level + 1).to(tl.float64)
            delta = high_width - low_width
            available = 0
            for start in tl.range(0, N, BLOCK_N):
                offset = start + lane
                active = offset < count
                current = tl.load(
                    OUT + row + offset, mask=offset < N, other=0,
                )
                available += tl.sum(
                    (active & (current == level)).to(tl.int32), axis=0,
                )
            requested = (under_gap / delta + 0.5).to(tl.int32)
            requested = tl.where(under_gap > 0.5, tl.maximum(requested, 0), 0)
            take_count = tl.minimum(available, requested)
            remaining = take_count
            while remaining > 0:
                threshold = -float("inf")
                for start in tl.range(0, N, BLOCK_N):
                    offset = start + lane
                    in_bounds = offset < N
                    active = offset < count
                    current = tl.load(
                        OUT + row + offset, mask=in_bounds, other=0,
                    )
                    score = tl.load(
                        SCORES + row + offset, mask=in_bounds, other=0.0,
                    ).to(tl.float32)
                    candidate = active & (current == level)
                    tile_max = tl.max(
                        tl.where(candidate, score, -float("inf")), axis=0,
                    )
                    threshold = tl.maximum(threshold, tile_max)

                group_count = 0
                # Descending flat offsets reproduce reverse-index tie repair.
                for reverse_chunk in tl.range(0, chunks):
                    start = (chunks - 1 - reverse_chunk) * BLOCK_N
                    offset = start + (BLOCK_N - 1 - lane)
                    in_bounds = offset < N
                    active = offset < count
                    current = tl.load(
                        OUT + row + offset, mask=in_bounds, other=0,
                    )
                    score = tl.load(
                        SCORES + row + offset, mask=in_bounds, other=0.0,
                    ).to(tl.float32)
                    equal = active & (current == level) & (score == threshold)
                    local_rank = tl.cumsum(equal.to(tl.int32), axis=0)
                    selected = equal & (local_rank + group_count <= remaining)
                    tl.store(
                        OUT + row + offset,
                        tl.where(selected, level + 1, current),
                        mask=in_bounds,
                    )
                    group_count += tl.sum(equal.to(tl.int32), axis=0)
                consumed = tl.minimum(group_count, remaining)
                remaining -= consumed
            under_gap -= take_count.to(tl.float64) * delta

        for start in tl.range(0, N, BLOCK_N):
            offset = start + lane
            in_bounds = offset < N
            current = tl.load(OUT + row + offset, mask=in_bounds, other=0)
            nominal = tl.load(NOMINAL_BITS + current)
            tl.store(OUT + row + offset, nominal, mask=in_bounds)

@dataclass(frozen=True)
class _DeviceConstants:
    """Small immutable tensors cached per allocator/device."""

    epsilon: torch.Tensor
    effective_bits: torch.Tensor
    nominal_bits: torch.Tensor


def _effective_bits(
    bits: int,
    n_outlier: int,
    head_dim: int,
    outlier_min_bits: int,
) -> float:
    """Mirror ``kvquant.allocator.effective_bits`` without hot-path imports."""
    if bits == 0 or bits >= 16 or n_outlier <= 0:
        return float(bits)
    outlier_bits = max(bits, outlier_min_bits)
    return bits + n_outlier * max(0, outlier_bits - bits) / head_dim


class BatchedNativeAllocator:
    """Allocate one layer for an entire serving batch on the score device.

    Parameters are immutable request-policy constants.  ``allocate`` accepts
    scores shaped ``[B, H_kv, T]`` and returns int32 bits of the same shape.
    The batch axis is never pooled: each row of ``B`` gets its own lambda and
    budget repair.

    ``return_lambda=True`` returns a device tensor of shape ``[B]``.  Keeping
    lambda on device is an intentional difference from the diagnostic reference
    API, which returns a Python scalar and therefore synchronizes execution.
    """

    def __init__(
        self,
        *,
        bit_levels: tuple[int, ...],
        target_avg_bits: float,
        epsilon: dict[int, float],
        sink_bits: int = 16,
        sink_tokens: int = 0,
        tail_tokens: int = 0,
        eviction_cost: float = 0.5,
        n_outlier: int = 0,
        head_dim: int = 128,
        outlier_min_bits: int = 4,
        above_target_alpha: float = 1.0,
        search_steps: int = 64,
    ) -> None:
        levels = tuple(sorted(set(int(level) for level in bit_levels)))
        if not levels:
            raise ValueError("bit_levels must contain at least one level")
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if search_steps <= 0:
            raise ValueError("search_steps must be positive")
        if sink_tokens < 0 or tail_tokens < 0:
            raise ValueError("sink_tokens and tail_tokens must be non-negative")

        self.bit_levels = levels
        self.target_avg_bits = float(target_avg_bits)
        self.sink_bits = int(sink_bits)
        self.sink_tokens = int(sink_tokens)
        self.tail_tokens = int(tail_tokens)
        self.eviction_cost = float(eviction_cost)
        self.n_outlier = int(n_outlier)
        self.head_dim = int(head_dim)
        self.outlier_min_bits = int(outlier_min_bits)
        self.above_target_alpha = float(above_target_alpha)
        self.search_steps = int(search_steps)

        target = max(float(levels[0]), min(float(levels[-1]), self.target_avg_bits))
        self._target = target

        eps = [float(epsilon.get(level, 0.0)) for level in levels]
        if levels[0] == 0 and self.eviction_cost != 1.0:
            eps[0] *= self.eviction_cost

        # Match the reference's optional compression of benefits above target.
        if self.above_target_alpha < 1.0:
            below = [
                level for level in levels
                if level <= self.target_avg_bits and level != 0
            ]
            eps_at_target = epsilon.get(int(self.target_avg_bits))
            if eps_at_target is None and below:
                eps_at_target = epsilon.get(below[-1], 0.0)
            if eps_at_target is not None and eps_at_target > 1e-12:
                for idx, level in enumerate(levels):
                    if level > self.target_avg_bits:
                        eps[idx] = (
                            (1.0 - self.above_target_alpha) * eps_at_target
                            + self.above_target_alpha * eps[idx]
                        )

        self._epsilon = tuple(eps)
        self._effective = tuple(
            _effective_bits(
                level,
                self.n_outlier,
                self.head_dim,
                self.outlier_min_bits,
            )
            for level in levels
        )
        # Tightest host-computable upper bound on the budget-hitting lambda*.  A
        # token switches from a higher level ``a`` to a lower level ``b`` at
        # ``lambda = score * (eps[b]-eps[a]) / (eff[a]-eff[b])``; beyond the
        # largest such breakpoint every token sits at the minimum level, so
        # ``lambda* <= max_active_score * slope_max``.  ``_initial_hi`` uses this
        # to replace the loose ``[0, 1e8]`` bracket when ``search_steps < 64``.
        # Host-side, over the K bit levels (constant, one-time at construction),
        # using only policy constants — not a per-request/device batch loop.
        slope_max = 0.0
        for hi_lvl in range(len(levels)):
            for lo_lvl in range(len(levels)):
                den = self._effective[hi_lvl] - self._effective[lo_lvl]
                num = self._epsilon[lo_lvl] - self._epsilon[hi_lvl]
                if den > 0.0 and num > 0.0:
                    slope_max = max(slope_max, num / den)
        self._slope_max = float(slope_max)
        self._constant_cache: dict[tuple[str, Optional[int]], _DeviceConstants] = {}

    def _constants(self, device: torch.device) -> _DeviceConstants:
        key = (device.type, device.index)
        cached = self._constant_cache.get(key)
        if cached is None:
            cached = _DeviceConstants(
                epsilon=torch.tensor(self._epsilon, dtype=torch.float32, device=device),
                effective_bits=torch.tensor(
                    self._effective, dtype=torch.float32, device=device,
                ),
                nominal_bits=torch.tensor(
                    self.bit_levels, dtype=torch.int32, device=device,
                ),
            )
            self._constant_cache[key] = cached
        return cached

    @staticmethod
    def _sanitize_scores(scores: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """Vectorized per-request equivalent of the reference sanitization."""
        scores = scores.float().clamp(min=0.0)
        finite_active = scores.isfinite() & active
        has_finite = finite_active.any(dim=1, keepdim=True)
        negative_inf = torch.full_like(scores, -torch.inf)
        max_finite = torch.where(finite_active, scores, negative_inf).amax(
            dim=1, keepdim=True,
        )

        # If a request has at least one finite active score, all non-finite
        # active values become its maximum finite value.  If it has none, retain
        # the reference fallback: NaN -> 0 and infinities -> dtype extrema.
        mixed = torch.where(scores.isfinite(), scores, max_finite)
        all_nonfinite = torch.nan_to_num(scores, nan=0.0)
        sanitized = torch.where(has_finite, mixed, all_nonfinite)
        return torch.where(active, sanitized, torch.zeros_like(sanitized))

    @staticmethod
    def _assign(
        scores: torch.Tensor,
        active: torch.Tensor,
        active_count: torch.Tensor,
        lam: torch.Tensor,
        constants: _DeviceConstants,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate all candidate levels for every request/cell."""
        cost = (
            scores[:, None, :] * constants.epsilon[None, :, None]
            + lam[:, None, None] * constants.effective_bits[None, :, None]
        )
        chosen = cost.argmin(dim=1)
        assigned_effective = constants.effective_bits[chosen]
        # ``allocate`` compacts active cells to the front of every row.  A
        # one-segment reduction with a device-resident length therefore has the
        # same numerical path as the reference's ``bits_t[chosen].mean()``,
        # including CUDA half-budget rounding, without a dynamic gather or a
        # host read.  ``unsafe=True`` permits the inactive padding after the
        # segment; it does not weaken bounds because active_count <= row width.
        active_total = torch.segment_reduce(
            assigned_effective,
            "sum",
            lengths=active_count[:, None],
            axis=1,
            unsafe=True,
        ).squeeze(1)
        denominator = active_count.clamp_min(1).to(torch.float32)
        # CUDA mean applies a rounded reciprocal multiplier; direct tensor
        # division differs by one ulp for counts such as 15 or 378.  CPU mean
        # follows direct division.  Branching on device metadata is asynchronous.
        if assigned_effective.is_cuda:
            average = active_total * denominator.reciprocal()
        else:
            average = active_total / denominator
        average = torch.where(active_count > 0, average, torch.zeros_like(average))
        return chosen, average

    def _initial_hi(
        self,
        scores: torch.Tensor,
        active_count: torch.Tensor,
    ) -> torch.Tensor:
        """Tightened per-request upper bound for the lambda bisection.

        ``lambda* <= max_active_score * slope_max``, so this bracket lets ~15
        bisections match the 64-step ``[0, 1e8]`` tags.  ``scores`` is sanitized
        (every cell ``>= 0``), so the plain row max is already an upper bound on
        the max active score — no front mask needed (an over-estimate only makes
        ``hi`` looser, which stays a valid upper bound).  Callers restrict this
        to the streaming path, where the bisection dominates; the persistent
        kernel keeps the frozen 64-step ``[0, 1e8]`` search (register-resident
        and cheaper than recomputing a bound).
        """
        batch = scores.shape[0]
        loose = torch.full(
            (batch,), 1.0e8, dtype=torch.float32, device=scores.device,
        )
        if not math.isfinite(self._slope_max) or self._slope_max <= 0.0:
            return loose
        smax = scores.float().amax(dim=1)
        tight = (smax * (self._slope_max * _HI_SAFETY)).clamp(min=1e-6)
        # Empty requests keep the harmless loose bracket (count==0 => avg==0).
        return torch.where(active_count > 0, tight, loose)

    def _search_lambda(
        self,
        scores: torch.Tensor,
        active: torch.Tensor,
        active_count: torch.Tensor,
        constants: _DeviceConstants,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Batched lambda search with a persistent Triton CUDA fast path.

        The persistent kernel (``row_width <= _MAX_PERSISTENT_CELLS``) is
        register-resident and already cheap at 64 steps, so it always keeps the
        frozen ``[0, 1e8]`` / 64-step search byte-for-byte.  The streaming kernel
        (wider rows / long context) is where the bisection cost dominates, so it
        honours the configured (possibly reduced) ``search_steps`` on the
        tightened bracket from ``_initial_hi``.
        """
        row_width = scores.shape[1]
        streaming = row_width > _MAX_PERSISTENT_CELLS
        # Fast path only where it pays off: the streaming search.  Persistent
        # rows stay at 64 steps / loose bracket (measured: reducing them there is
        # net-negative once the bound recompute is priced in).
        if streaming and self.search_steps < 64:
            steps = self.search_steps
            hi_init = self._initial_hi(scores, active_count)
        else:
            steps = 64
            hi_init = torch.full(
                (scores.shape[0],), 1.0e8, dtype=torch.float32, device=scores.device,
            )
        # With no outlier-channel surcharge, every effective width is an
        # integer.  A persistent request-level CTA can therefore reproduce the
        # fp32 budget comparison exactly while replacing ~65 expensive
        # segment-reduce launches with one launch.  OCS keeps the conservative
        # eager backend because its fractional effective widths are sensitive
        # to reduction order.  Very wide rows also fall back to avoid excessive
        # register pressure in a single CTA.
        if _HAS_TRITON and scores.is_cuda and self.n_outlier == 0:
            lam = torch.empty(
                scores.shape[0], dtype=torch.float32, device=scores.device,
            )
            if not streaming:
                block_n = max(32, triton.next_power_of_2(row_width))
                _persistent_lambda_search_kernel[(scores.shape[0],)](
                    scores,
                    active_count,
                    constants.epsilon,
                    constants.effective_bits,
                    lam,
                    hi_init,
                    N=row_width,
                    NUM_LEVELS=len(self.bit_levels),
                    TARGET=self._target,
                    SEARCH_STEPS=steps,
                    BLOCK_N=block_n,
                    num_warps=8,
                )
            else:
                _streaming_lambda_search_kernel[(scores.shape[0],)](
                    scores,
                    active_count,
                    constants.epsilon,
                    constants.effective_bits,
                    lam,
                    hi_init,
                    row_width,
                    NUM_LEVELS=len(self.bit_levels),
                    TARGET=self._target,
                    SEARCH_STEPS=steps,
                    BLOCK_N=1024,
                    num_warps=8,
                )
            # Re-evaluate the final boundary with the exact eager arithmetic
            # used by the semantic reference.  Triton may contract the last
            # multiply-add into an FMA; at a breakpoint that can change one
            # interchangeable cell even when lambda is bit-identical.  This
            # single assignment/segment reduction preserves exact identities
            # while the 64 search reductions remain fused above.
            chosen, average = self._assign(
                scores, active, active_count, lam, constants,
            )
            return chosen, average, lam

        # Correctness fallback for CPU, OCS, and unusually wide rows.
        batch = scores.shape[0]
        lo = torch.zeros(batch, dtype=torch.float32, device=scores.device)
        hi = hi_init.clone()
        target = torch.full_like(lo, self._target)
        for _ in range(steps):
            mid = (lo + hi) * 0.5
            _, average = self._assign(scores, active, active_count, mid, constants)
            over = average > target
            lo = torch.where(over, mid, lo)
            hi = torch.where(over, hi, mid)

        lam = (lo + hi) * 0.5
        chosen, average = self._assign(scores, active, active_count, lam, constants)
        return chosen, average, lam

    def _repair_budget(
        self,
        scores: torch.Tensor,
        active: torch.Tensor,
        active_count: torch.Tensor,
        chosen: torch.Tensor,
        average: torch.Tensor,
        constants: _DeviceConstants,
    ) -> torch.Tensor:
        """Device-only, batched form of the reference's level-ordered repair."""
        if _HAS_TRITON and scores.is_cuda and self.n_outlier == 0:
            row_width = scores.shape[1]
            repaired = torch.empty_like(scores, dtype=torch.int32)
            if row_width <= _MAX_PERSISTENT_CELLS:
                block_n = max(32, triton.next_power_of_2(row_width))
                _persistent_budget_repair_kernel[(scores.shape[0],)](
                    scores,
                    active_count,
                    chosen,
                    average,
                    constants.effective_bits,
                    constants.nominal_bits,
                    repaired,
                    N=row_width,
                    NUM_LEVELS=len(self.bit_levels),
                    TARGET=self._target,
                    BLOCK_N=block_n,
                    num_warps=8,
                )
            else:
                _streaming_budget_repair_kernel[(scores.shape[0],)](
                    scores,
                    active_count,
                    chosen,
                    average,
                    constants.effective_bits,
                    constants.nominal_bits,
                    repaired,
                    row_width,
                    NUM_LEVELS=len(self.bit_levels),
                    TARGET=self._target,
                    BLOCK_N=1024,
                    num_warps=8,
                )
            return repaired

        effective = constants.effective_bits[chosen]
        nominal = constants.nominal_bits[chosen]

        # The reference converts the fp32 average to a Python float before gap
        # arithmetic.  fp64 device math reproduces that arithmetic closely while
        # retaining an asynchronous execution path.
        gap = (
            average.to(torch.float64) - float(self._target)
        ) * active_count.to(torch.float64)

        # Sinks are not part of the reference's compact active vector.  Sorting
        # them after all finite active scores gives the same active ordering
        # without a dynamic gather per request.
        sort_key = torch.where(active, scores, torch.full_like(scores, torch.inf))
        # Explicit stable ordering gives a reproducible policy for equal scores.
        # The diagnostic reference leaves this unspecified (stable=False).
        order = torch.argsort(sort_key, dim=1, stable=True)
        active_ordered = active.gather(1, order)
        effective_ordered = effective.gather(1, order)
        nominal_ordered = nominal.gather(1, order)

        over_gap = gap
        for idx in range(len(self.bit_levels) - 1):
            effective_high = float(self._effective[idx + 1])
            effective_low = float(self._effective[idx])
            delta = effective_high - effective_low
            if delta <= 0.0:
                continue
            at_high = active_ordered & (effective_ordered == effective_high)
            available = at_high.sum(dim=1, dtype=torch.int64)
            requested = torch.trunc(over_gap / delta + 0.5).to(torch.int64)
            requested = torch.where(
                over_gap > 0.5,
                requested.clamp_min(0),
                torch.zeros_like(requested),
            )
            count = torch.minimum(available, requested)
            rank = at_high.cumsum(dim=1)
            selected = at_high & (rank <= count[:, None])
            effective_ordered = torch.where(
                selected,
                torch.full_like(effective_ordered, effective_low),
                effective_ordered,
            )
            nominal_ordered = torch.where(
                selected,
                torch.full_like(nominal_ordered, self.bit_levels[idx]),
                nominal_ordered,
            )
            over_gap = over_gap - count.to(torch.float64) * delta

        under_gap = -gap
        for idx in range(len(self.bit_levels) - 2, -1, -1):
            effective_low = float(self._effective[idx])
            effective_high = float(self._effective[idx + 1])
            delta = effective_high - effective_low
            if delta <= 0.0:
                continue
            at_low = active_ordered & (effective_ordered == effective_low)
            available = at_low.sum(dim=1, dtype=torch.int64)
            requested = torch.trunc(under_gap / delta + 0.5).to(torch.int64)
            requested = torch.where(
                under_gap > 0.5,
                requested.clamp_min(0),
                torch.zeros_like(requested),
            )
            count = torch.minimum(available, requested)
            reverse_rank = at_low.flip(1).cumsum(dim=1).flip(1)
            selected = at_low & (reverse_rank <= count[:, None])
            effective_ordered = torch.where(
                selected,
                torch.full_like(effective_ordered, effective_high),
                effective_ordered,
            )
            nominal_ordered = torch.where(
                selected,
                torch.full_like(nominal_ordered, self.bit_levels[idx + 1]),
                nominal_ordered,
            )
            under_gap = under_gap - count.to(torch.float64) * delta

        repaired = torch.empty_like(nominal_ordered)
        repaired.scatter_(1, order, nominal_ordered)
        return repaired

    @torch.no_grad()
    def allocate(
        self,
        scores: torch.Tensor,
        *,
        sink_mask: Optional[torch.Tensor] = None,
        tail_tokens: Optional[int] = None,
        fixed_lambda: Optional[float | torch.Tensor] = None,
        return_lambda: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return per-cell bit levels for ``scores[B,H_kv,T]``.

        ``fixed_lambda`` mirrors the reference's streaming mode and therefore
        skips local budget repair.  A tensor lambda may contain either one value
        or one value per request.
        """
        if scores.dim() != 3:
            raise ValueError(
                f"expected scores [B,H_kv,T], got shape {tuple(scores.shape)}"
            )
        batch, heads, tokens = scores.shape
        device = scores.device
        constants = self._constants(device)

        # Match the reference's empty-active behavior without asking a
        # reduction such as ``amax`` to operate on an empty flattened axis.
        # Shape metadata is host-side, so this branch does not synchronize the
        # score device.
        if heads == 0 or tokens == 0:
            bits = torch.empty_like(scores, dtype=torch.int32)
            lam = torch.zeros(batch, dtype=torch.float32, device=device)
            return (bits, lam) if return_lambda else bits

        if sink_mask is None:
            positions = torch.arange(tokens, device=device)
            base_sink = positions < min(self.sink_tokens, tokens)
            active_3d = ~base_sink.view(1, 1, tokens).expand(batch, heads, tokens)
        else:
            if sink_mask.shape != scores.shape:
                raise ValueError(
                    f"sink_mask shape {tuple(sink_mask.shape)} does not match "
                    f"scores {tuple(scores.shape)}"
                )
            active_3d = ~sink_mask.to(device=device, dtype=torch.bool)

        active = active_3d.reshape(batch, -1)
        active_count = active.sum(dim=1, dtype=torch.int64)
        flat_scores = self._sanitize_scores(scores.reshape(batch, -1), active)

        # Compact active cells before lambda search.  ``destination`` maps each
        # original flat position to its packed position: active cells retain
        # their original relative order at the front and sinks retain theirs in
        # the padding.  This cumsum/scatter construction is O(B*N), unlike a
        # second argsort, and all indices remain on the score device.
        active_rank = active.cumsum(dim=1, dtype=torch.int64) - 1
        inactive = ~active
        inactive_rank = inactive.cumsum(dim=1, dtype=torch.int64) - 1
        destination = torch.where(
            active,
            active_rank,
            active_count[:, None] + inactive_rank,
        )
        packed_scores = torch.empty_like(flat_scores)
        packed_scores.scatter_(1, destination, flat_scores)
        packed_positions = torch.arange(
            flat_scores.shape[1], device=device,
        ).view(1, -1)
        packed_active = packed_positions < active_count[:, None]

        if len(self.bit_levels) == 1:
            bits = torch.full(
                packed_scores.shape,
                self.bit_levels[0],
                dtype=torch.int32,
                device=device,
            )
            lam = torch.zeros(batch, dtype=torch.float32, device=device)
        elif fixed_lambda is None:
            chosen, average, lam = self._search_lambda(
                packed_scores, packed_active, active_count, constants,
            )
            bits = self._repair_budget(
                packed_scores,
                packed_active,
                active_count,
                chosen,
                average,
                constants,
            )
        else:
            lam = torch.as_tensor(fixed_lambda, dtype=torch.float32, device=device)
            lam = lam.reshape(-1).expand(batch)
            chosen, _ = self._assign(
                packed_scores, packed_active, active_count, lam, constants,
            )
            bits = constants.nominal_bits[chosen]

        bits = torch.where(
            packed_active,
            bits,
            torch.full_like(bits, self.sink_bits),
        )
        # ``destination[original] = packed``; gathering by it restores the
        # original (head, token) positions without an inverse sort.
        bits = bits.gather(1, destination).reshape_as(scores)

        protected_tail = self.tail_tokens if tail_tokens is None else int(tail_tokens)
        if protected_tail > 0 and tokens > 0:
            width = min(protected_tail, tokens)
            bits = bits.clone()
            bits[..., -width:] = self.sink_bits

        lam = torch.where(
            active_count > 0,
            lam,
            torch.zeros_like(lam),
        )
        return (bits, lam) if return_lambda else bits


def native_optimal_scores_to_bits(
    scores: torch.Tensor,
    *,
    bit_levels: tuple[int, ...],
    target_avg_bits: float,
    epsilon: dict[int, float],
    sink_mask: Optional[torch.Tensor] = None,
    sink_bits: int = 16,
    sink_tokens: int = 0,
    tail_tokens: int = 0,
    eviction_cost: float = 0.5,
    n_outlier: int = 0,
    head_dim: int = 128,
    outlier_min_bits: int = 4,
    above_target_alpha: float = 1.0,
    fixed_lambda: Optional[float | torch.Tensor] = None,
    return_lambda: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Stateless convenience wrapper; reuse the class on a serving hot path."""
    allocator = BatchedNativeAllocator(
        bit_levels=bit_levels,
        target_avg_bits=target_avg_bits,
        epsilon=epsilon,
        sink_bits=sink_bits,
        sink_tokens=sink_tokens,
        tail_tokens=tail_tokens,
        eviction_cost=eviction_cost,
        n_outlier=n_outlier,
        head_dim=head_dim,
        outlier_min_bits=outlier_min_bits,
        above_target_alpha=above_target_alpha,
    )
    return allocator.allocate(
        scores,
        sink_mask=sink_mask,
        fixed_lambda=fixed_lambda,
        return_lambda=return_lambda,
    )


__all__ = ["BatchedNativeAllocator", "native_optimal_scores_to_bits"]
