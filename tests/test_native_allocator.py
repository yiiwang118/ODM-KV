from __future__ import annotations

import inspect

import pytest
import torch

from kvquant.allocator import optimal_scores_to_bits
import kvquant.runtime.native_allocator as native_allocator_module
from kvquant.runtime.native_allocator import (
    BatchedNativeAllocator,
    native_optimal_scores_to_bits,
)


EPSILON = {
    0: 1.0,
    2: 0.116,
    3: 0.034,
    4: 0.0093,
    8: 4.9e-5,
    16: 0.0,
}


def _oracle_batch(
    scores: torch.Tensor,
    *,
    bit_levels: tuple[int, ...],
    target: float,
    sink_mask: torch.Tensor | None,
    sink_bits: int = 16,
    tail_tokens: int = 0,
    eviction_cost: float = 0.5,
    above_target_alpha: float = 1.0,
    n_outlier: int = 0,
    head_dim: int = 128,
    outlier_min_bits: int = 4,
) -> torch.Tensor:
    allocated = []
    for request in range(scores.shape[0]):
        request_sink = None if sink_mask is None else sink_mask[request]
        allocated.append(
            optimal_scores_to_bits(
                scores[request],
                bit_levels=bit_levels,
                target_avg_bits=target,
                epsilon=EPSILON,
                sink_mask=request_sink,
                sink_bits=sink_bits,
                eviction_cost=eviction_cost,
                above_target_alpha=above_target_alpha,
                n_outlier=n_outlier,
                head_dim=head_dim,
                outlier_min_bits=outlier_min_bits,
            )
        )
    bits = torch.stack(allocated)
    if tail_tokens > 0:
        bits = bits.clone()
        bits[..., -min(tail_tokens, bits.shape[-1]):] = sink_bits
    return bits


def _devices() -> list[str]:
    return ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("batch", [1, 8])
@pytest.mark.parametrize("tokens", [7, 31, 128])
@pytest.mark.parametrize(
    ("levels", "target", "alpha"),
    [
        ((0, 2, 3, 4, 8, 16), 1.0, 1.0),
        ((0, 2, 4, 8), 2.5, 0.5),
        ((2, 4, 8, 16), 3.0, 1.0),
        ((0, 4, 16), 5.0, 1.0),
        ((4,), 4.0, 1.0),
    ],
)
def test_matches_reference_random(
    device: str,
    batch: int,
    tokens: int,
    levels: tuple[int, ...],
    target: float,
    alpha: float,
) -> None:
    generator = torch.Generator(device=device).manual_seed(1701 + batch + tokens)
    scores = torch.rand(batch, 3, tokens, generator=generator, device=device)
    sink = torch.zeros_like(scores, dtype=torch.bool)
    sink[..., : min(2, tokens)] = True

    allocator = BatchedNativeAllocator(
        bit_levels=levels,
        target_avg_bits=target,
        epsilon=EPSILON,
        sink_bits=16,
        eviction_cost=0.5,
        above_target_alpha=alpha,
    )
    got = allocator.allocate(scores, sink_mask=sink)
    expected = _oracle_batch(
        scores,
        bit_levels=levels,
        target=target,
        sink_mask=sink,
        eviction_cost=0.5,
        above_target_alpha=alpha,
    )
    assert torch.equal(got, expected)


@pytest.mark.parametrize("device", _devices())
@pytest.mark.parametrize("batch", [1, 8])
def test_ties_preserve_histogram_and_native_is_deterministic(
    device: str, batch: int,
) -> None:
    scores = torch.full((batch, 4, 33), 0.25, device=device)
    scores[:, :, ::7] = 1.0
    scores[:, :, 1::9] = 0.0
    sink = torch.zeros_like(scores, dtype=torch.bool)
    sink[..., :4] = True

    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 3, 4, 8, 16),
        target_avg_bits=1.0,
        epsilon=EPSILON,
        eviction_cost=0.5,
    )
    got = allocator.allocate(scores, sink_mask=sink)
    repeated = allocator.allocate(scores, sink_mask=sink)
    expected = _oracle_batch(
        scores,
        bit_levels=(0, 2, 3, 4, 8, 16),
        target=1.0,
        sink_mask=sink,
    )
    assert torch.equal(got, repeated)

    # The reference repair calls argsort with stable=False.  Equal-score cells
    # are interchangeable in its objective, and their exact promoted identities
    # vary with sort backend/shape.  Native defines a stable flat-index policy;
    # the allocation histogram and therefore the budget remain exactly equal.
    for request in range(batch):
        for level in (0, 2, 3, 4, 8, 16):
            assert torch.equal(
                (got[request] == level).sum(),
                (expected[request] == level).sum(),
            )


@pytest.mark.parametrize("device", _devices())
def test_nan_inf_sanitization_matches_reference(device: str) -> None:
    scores = torch.tensor(
        [
            [[float("nan"), float("inf"), -float("inf"), -2.0, 0.2, 0.8]],
            [[float("nan"), float("inf"), float("nan"), float("inf"), 0.1, 0.9]],
            [[float("nan"), float("inf"), float("nan"), float("inf"), float("nan"), float("inf")]],
        ],
        device=device,
    )
    sink = torch.zeros_like(scores, dtype=torch.bool)
    sink[..., 0] = True
    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 4, 8, 16),
        target_avg_bits=2.0,
        epsilon=EPSILON,
    )
    got = allocator.allocate(scores, sink_mask=sink)
    expected = _oracle_batch(
        scores,
        bit_levels=(0, 2, 4, 8, 16),
        target=2.0,
        sink_mask=sink,
    )
    repeated = allocator.allocate(scores, sink_mask=sink)
    assert torch.equal(got, repeated)
    # Sanitization deliberately maps several NaN/Inf cells to the same finite
    # request maximum.  Their identities are therefore another exact-score tie:
    # native's stable policy may differ from the reference CUDA argsort, while
    # the allocation histogram and budget must remain identical.
    for request in range(scores.shape[0]):
        for level in (0, 2, 4, 8, 16):
            assert torch.equal(
                (got[request] == level).sum(),
                (expected[request] == level).sum(),
            )


@pytest.mark.parametrize("device", _devices())
def test_sink_and_tail_override_order_matches_press(device: str) -> None:
    scores = torch.linspace(0, 1, 2 * 3 * 20, device=device).reshape(2, 3, 20)
    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 3, 4, 8, 16),
        target_avg_bits=1.0,
        epsilon=EPSILON,
        sink_tokens=4,
        tail_tokens=7,
        eviction_cost=0.5,
    )
    got = allocator.allocate(scores)
    sink = torch.zeros_like(scores, dtype=torch.bool)
    sink[..., :4] = True
    expected = _oracle_batch(
        scores,
        bit_levels=(0, 2, 3, 4, 8, 16),
        target=1.0,
        sink_mask=sink,
        tail_tokens=7,
    )
    assert torch.equal(got, expected)
    assert torch.all(got[..., :4] == 16)
    assert torch.all(got[..., -7:] == 16)


@pytest.mark.parametrize("device", _devices())
def test_each_request_has_an_independent_budget(device: str) -> None:
    generator = torch.Generator(device=device).manual_seed(904)
    scores = torch.rand(8, 4, 97, generator=generator, device=device)
    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 3, 4, 8, 16),
        target_avg_bits=1.0,
        epsilon=EPSILON,
        sink_tokens=4,
        eviction_cost=0.5,
    )
    before = allocator.allocate(scores)
    changed = scores.clone()
    changed[0].mul_(0).add_(1000)
    after = allocator.allocate(changed)
    assert torch.equal(before[1:], after[1:])


@pytest.mark.parametrize("device", _devices())
def test_eviction_cost_and_above_target_semantics(device: str) -> None:
    generator = torch.Generator(device=device).manual_seed(441)
    scores = torch.rand(4, 3, 101, generator=generator, device=device)
    for eviction_cost, alpha in ((0.5, 1.0), (1.0, 1.0), (0.5, 0.25)):
        allocator = BatchedNativeAllocator(
            bit_levels=(0, 2, 3, 4, 8, 16),
            target_avg_bits=3.0,
            epsilon=EPSILON,
            sink_tokens=3,
            eviction_cost=eviction_cost,
            above_target_alpha=alpha,
        )
        got = allocator.allocate(scores)
        sink = torch.zeros_like(scores, dtype=torch.bool)
        sink[..., :3] = True
        expected = _oracle_batch(
            scores,
            bit_levels=(0, 2, 3, 4, 8, 16),
            target=3.0,
            sink_mask=sink,
            eviction_cost=eviction_cost,
            above_target_alpha=alpha,
        )
        assert torch.equal(got, expected)


@pytest.mark.parametrize("device", _devices())
def test_ocs_effective_budget_matches_reference(device: str) -> None:
    scores = torch.linspace(0.0, 1.0, 2 * 2 * 79, device=device).reshape(2, 2, 79)
    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 4, 8, 16),
        target_avg_bits=2.0,
        epsilon=EPSILON,
        sink_tokens=2,
        eviction_cost=0.5,
        n_outlier=8,
        head_dim=128,
        outlier_min_bits=4,
    )
    got = allocator.allocate(scores)
    sink = torch.zeros_like(scores, dtype=torch.bool)
    sink[..., :2] = True
    expected = _oracle_batch(
        scores,
        bit_levels=(0, 2, 4, 8, 16),
        target=2.0,
        sink_mask=sink,
        n_outlier=8,
        head_dim=128,
        outlier_min_bits=4,
    )
    assert torch.equal(got, expected)


def test_all_sink_and_short_tail_are_well_defined() -> None:
    scores = torch.rand(3, 2, 4)
    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 16),
        target_avg_bits=1.0,
        epsilon=EPSILON,
        sink_tokens=99,
        tail_tokens=99,
    )
    bits, lam = allocator.allocate(scores, return_lambda=True)
    assert torch.all(bits == 16)
    assert torch.equal(lam, torch.zeros_like(lam))


@pytest.mark.parametrize("shape", [(3, 2, 0), (3, 0, 11)])
def test_empty_active_axis_is_well_defined(shape: tuple[int, int, int]) -> None:
    scores = torch.empty(shape)
    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 16),
        target_avg_bits=1.0,
        epsilon=EPSILON,
    )
    bits, lam = allocator.allocate(scores, return_lambda=True)
    assert bits.shape == scores.shape
    assert bits.dtype == torch.int32
    assert torch.equal(lam, torch.zeros_like(lam))


@pytest.mark.parametrize("device", _devices())
def test_fixed_lambda_matches_reference_without_budget_repair(device: str) -> None:
    generator = torch.Generator(device=device).manual_seed(118)
    scores = torch.rand(8, 3, 73, generator=generator, device=device)
    sink = torch.zeros_like(scores, dtype=torch.bool)
    sink[..., :3] = True
    fixed_lambda = 0.013
    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 3, 4, 8, 16),
        target_avg_bits=1.0,
        epsilon=EPSILON,
        eviction_cost=0.5,
    )
    got, lam = allocator.allocate(
        scores,
        sink_mask=sink,
        fixed_lambda=fixed_lambda,
        return_lambda=True,
    )
    expected = torch.stack([
        optimal_scores_to_bits(
            scores[request],
            bit_levels=(0, 2, 3, 4, 8, 16),
            target_avg_bits=1.0,
            epsilon=EPSILON,
            sink_mask=sink[request],
            eviction_cost=0.5,
            fixed_lambda=fixed_lambda,
        )
        for request in range(scores.shape[0])
    ])
    assert torch.equal(got, expected)
    assert torch.equal(lam, torch.full_like(lam, fixed_lambda))


def test_stateless_wrapper_matches_reusable_allocator() -> None:
    scores = torch.rand(2, 3, 27)
    expected = BatchedNativeAllocator(
        bit_levels=(0, 2, 4, 16),
        target_avg_bits=2.0,
        epsilon=EPSILON,
        sink_tokens=2,
        tail_tokens=5,
    ).allocate(scores)
    got = native_optimal_scores_to_bits(
        scores,
        bit_levels=(0, 2, 4, 16),
        target_avg_bits=2.0,
        epsilon=EPSILON,
        sink_tokens=2,
        tail_tokens=5,
    )
    assert torch.equal(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_production_shape_matches_reference() -> None:
    batch, heads, tokens = 8, 8, 2048
    base = torch.linspace(0.0, 1.0, heads * tokens, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(221)
    rows = []
    for request in range(batch):
        permutation = torch.randperm(
            heads * tokens, generator=generator, device="cuda",
        )
        rows.append((base[permutation] + request * 0.01).reshape(heads, tokens))
    scores = torch.stack(rows)
    sink = torch.zeros_like(scores, dtype=torch.bool)
    sink[..., :4] = True
    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 3, 4, 8, 16),
        target_avg_bits=1.0,
        epsilon=EPSILON,
        eviction_cost=0.5,
    )
    got = allocator.allocate(scores, sink_mask=sink)
    expected = _oracle_batch(
        scores,
        bit_levels=(0, 2, 3, 4, 8, 16),
        target=1.0,
        sink_mask=sink,
    )
    assert torch.equal(got, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_production_path_does_not_synchronize_to_host() -> None:
    scores = torch.rand(8, 8, 1024, device="cuda")
    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 3, 4, 8, 16),
        target_avg_bits=1.0,
        epsilon=EPSILON,
        sink_tokens=4,
        tail_tokens=128,
        eviction_cost=0.5,
    )
    allocator.allocate(scores)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        bits, lam = allocator.allocate(scores, return_lambda=True)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()
    assert bits.is_cuda
    assert lam.is_cuda


def test_production_source_has_no_host_value_reads_or_batch_loop() -> None:
    source = inspect.getsource(__import__(
        "kvquant.runtime.native_allocator", fromlist=["BatchedNativeAllocator"],
    ))
    forbidden = (".item" + "(", ".tolist" + "(", "for b in range")
    for marker in forbidden:
        assert marker not in source


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_persistent_kernels_match_device_fallback_on_hard_distributions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stress more allocation breakpoints and ties than the standard oracle."""
    batch, heads, tokens = 8, 8, 2048
    generator = torch.Generator(device="cuda").manual_seed(6197)
    random_scores = torch.rand(
        batch, heads, tokens, generator=generator, device="cuda",
    )
    dynamic_base = torch.logspace(
        -5, 5, heads * tokens, device="cuda", dtype=torch.float32,
    )
    dynamic_scores = torch.stack([
        dynamic_base.roll(997 * request).reshape(heads, tokens)
        for request in range(batch)
    ])
    tied_scores = torch.round(random_scores * 16.0) / 16.0

    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 3, 4, 8, 16),
        target_avg_bits=1.0,
        epsilon=EPSILON,
        sink_tokens=4,
        tail_tokens=128,
        eviction_cost=0.5,
    )
    for scores in (random_scores, dynamic_scores, tied_scores):
        fast = allocator.allocate(scores)
        with monkeypatch.context() as context:
            context.setattr(native_allocator_module, "_HAS_TRITON", False)
            fallback = allocator.allocate(scores)
        assert torch.equal(fast, fallback)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_all_sink_fast_path_returns_zero_lambda() -> None:
    scores = torch.rand(8, 8, 2048, device="cuda")
    sink = torch.ones_like(scores, dtype=torch.bool)
    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 3, 4, 8, 16),
        target_avg_bits=1.0,
        epsilon=EPSILON,
        eviction_cost=0.5,
    )
    bits, lam = allocator.allocate(scores, sink_mask=sink, return_lambda=True)
    assert torch.all(bits == 16)
    assert torch.equal(lam, torch.zeros_like(lam))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_streaming_long_row_matches_exact_device_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fixed-tile path must stay exact beyond the 32K resident limit."""
    generator = torch.Generator(device="cuda").manual_seed(8192)
    scores = torch.rand(8, 8, 8192, generator=generator, device="cuda")
    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 3, 4, 8, 16),
        target_avg_bits=1.0,
        epsilon=EPSILON,
        sink_tokens=4,
        tail_tokens=128,
        eviction_cost=0.5,
    )
    fast = allocator.allocate(scores)
    with monkeypatch.context() as context:
        context.setattr(native_allocator_module, "_HAS_TRITON", False)
        fallback = allocator.allocate(scores)
    assert torch.equal(fast, fallback)

    allocator.allocate(scores)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        bits, lam = allocator.allocate(scores, return_lambda=True)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert bits.is_cuda
    assert lam.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_streaming_long_row_ties_nonfinite_and_all_sink(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(16384)
    base = torch.rand(1, 8, 8192, generator=generator, device="cuda")
    tied = torch.round(base * 16.0) / 16.0
    nonfinite = base.clone()
    flat = nonfinite.view(-1)
    flat[::97] = torch.nan
    flat[1::131] = torch.inf
    flat[2::193] = -torch.inf
    allocator = BatchedNativeAllocator(
        bit_levels=(0, 2, 3, 4, 8, 16),
        target_avg_bits=1.0,
        epsilon=EPSILON,
        sink_tokens=4,
        tail_tokens=128,
        eviction_cost=0.5,
    )
    for scores in (tied, nonfinite):
        fast = allocator.allocate(scores)
        with monkeypatch.context() as context:
            context.setattr(native_allocator_module, "_HAS_TRITON", False)
            fallback = allocator.allocate(scores)
        assert torch.equal(fast, fallback)

    all_sink = torch.ones_like(base, dtype=torch.bool)
    bits, lam = allocator.allocate(base, sink_mask=all_sink, return_lambda=True)
    assert torch.all(bits == 16)
    assert torch.equal(lam, torch.zeros_like(lam))
