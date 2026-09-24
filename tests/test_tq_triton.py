"""Tests for kvquant.tq_triton.mse_nearest_centroid.

Invariants:
  1. Output matches the PyTorch searchsorted reference bit-for-bit (up to
     tolerances given by fp32 arithmetic — should actually be bit-exact
     since both use exp/abs/compare on the same fp32 inputs).
  2. Works on CPU (falls back to PyTorch oracle).
  3. Works on any CUDA device (including cuda:1 when visible).
  4. Works for bit levels from 1 to 8 (K from 2 to 256 centroids).
  5. TurboQuantMSE end-to-end produces identical results after the change.
"""
from __future__ import annotations

import pytest
import torch

from kvquant.tq_backend import TurboQuantMSE, _pack_indices
from kvquant.tq_triton import (
    HAS_TRITON,
    _nearest_centroid_reference,
    mse_nearest_centroid,
)


# ---------------------------------------------------------------------------
# Reference vs public wrapper on CPU
# ---------------------------------------------------------------------------

class TestNearestCentroidCPU:
    def test_shape_preserved(self):
        torch.manual_seed(0)
        y = torch.randn(3, 5, 7)
        centroids = torch.linspace(-1.0, 1.0, 4)
        out = mse_nearest_centroid(y, centroids)
        assert out.shape == y.shape
        assert out.dtype == torch.int32

    def test_matches_pytorch_reference(self):
        torch.manual_seed(0)
        y = torch.randn(10, 128)
        for bits in (1, 2, 3, 4, 5, 6, 7, 8):
            K = 2 ** bits
            centroids = torch.sort(torch.randn(K)).values
            got = mse_nearest_centroid(y, centroids)
            ref = _nearest_centroid_reference(y, centroids)
            assert torch.equal(got, ref), f"bits={bits} mismatch"

    def test_edge_cases(self):
        # y values at the extremes and exact centroid hits
        centroids = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        y = torch.tensor([-100.0, -1.0, -0.5, 0.25, 0.5, 0.6, 10.0])
        got = mse_nearest_centroid(y, centroids)
        ref = _nearest_centroid_reference(y, centroids)
        assert torch.equal(got, ref)

    def test_rejects_degenerate_codebook(self):
        with pytest.raises(ValueError, match="K>=2"):
            mse_nearest_centroid(torch.tensor([0.0]), torch.tensor([0.0]))

    def test_nonfinite_values_match_frozen_linear_scan(self):
        centroids = torch.tensor([-1.0, 0.0, 0.5, 1.0])
        values = torch.tensor([float("-inf"), float("inf"), float("nan")])
        got = mse_nearest_centroid(values, centroids)
        assert torch.equal(got, torch.zeros(3, dtype=torch.int32))


# ---------------------------------------------------------------------------
# Triton GPU path
# ---------------------------------------------------------------------------

CUDA_OK = HAS_TRITON and torch.cuda.is_available()
cuda_only = pytest.mark.skipif(not CUDA_OK, reason="needs CUDA + triton")


@cuda_only
class TestNearestCentroidCUDA:
    @pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_matches_reference(self, bits, dtype):
        torch.manual_seed(42)
        K = 2 ** bits
        y = torch.randn(17, 128, dtype=dtype, device="cuda") * 0.3
        centroids = torch.sort(torch.randn(K, device="cuda").float()).values

        got = mse_nearest_centroid(y, centroids)
        ref = _nearest_centroid_reference(y, centroids)
        # Both cast y to fp32 internally — should be identical.
        assert torch.equal(got, ref), f"bits={bits} dtype={dtype}"

    def test_shape_preserved(self):
        y = torch.randn(4, 8, 16, 128, device="cuda")
        centroids = torch.linspace(-1.0, 1.0, 16, device="cuda")
        out = mse_nearest_centroid(y, centroids)
        assert out.shape == y.shape
        assert out.device == y.device
        assert out.dtype == torch.int32

    def test_deterministic(self):
        torch.manual_seed(0)
        y = torch.randn(100, 128, device="cuda")
        centroids = torch.linspace(-1.0, 1.0, 16, device="cuda")
        a = mse_nearest_centroid(y, centroids)
        b = mse_nearest_centroid(y, centroids)
        assert torch.equal(a, b)

    @pytest.mark.parametrize("bits", [2, 3, 4, 8])
    def test_production_levels_exact_hits_midpoint_ties_and_extremes(self, bits):
        """Exercise the decisions most likely to differ from a linear scan.

        Midpoints must choose the lower centroid, exact hits must preserve the
        exact code, and finite values beyond the codebook must select an edge.
        This covers every quantized level in the production six-tag system.
        """
        k = 1 << bits
        centroids = torch.arange(k, device="cuda", dtype=torch.float32) * 0.25 - 1.0
        midpoints = (centroids[:-1] + centroids[1:]) * 0.5
        y = torch.cat(
            (
                centroids,
                midpoints,
                centroids[:1] - 100.0,
                centroids[-1:] + 100.0,
            )
        )
        expected = torch.cat(
            (
                torch.arange(k, device="cuda", dtype=torch.int32),
                torch.arange(k - 1, device="cuda", dtype=torch.int32),
                torch.tensor([0, k - 1], device="cuda", dtype=torch.int32),
            )
        )
        got = mse_nearest_centroid(y, centroids)
        assert torch.equal(got, expected), f"bits={bits}"

    def test_nonfinite_values_match_frozen_linear_scan(self):
        centroids = torch.tensor([-1.0, 0.0, 0.5, 1.0], device="cuda")
        values = torch.tensor(
            [float("-inf"), float("inf"), float("nan")], device="cuda",
        )
        got = mse_nearest_centroid(values, centroids)
        assert torch.equal(got, torch.zeros(3, device="cuda", dtype=torch.int32))


# ---------------------------------------------------------------------------
# Multi-GPU (device_map="auto" regression) — skipped if < 2 CUDA devices
# ---------------------------------------------------------------------------

multi_gpu_only = pytest.mark.skipif(
    not (CUDA_OK and torch.cuda.device_count() >= 2),
    reason="needs 2+ CUDA devices",
)


@multi_gpu_only
def test_works_on_non_default_device():
    """Kernel must launch on tensor's device, not just cuda:0."""
    torch.manual_seed(1)
    centroids = torch.linspace(-1.0, 1.0, 4).cuda()
    # Place input on cuda:1 while default context is cuda:0
    torch.cuda.set_device(0)
    y = torch.randn(100, 128, device="cuda:1")
    c = centroids.to("cuda:1")
    out = mse_nearest_centroid(y, c)
    assert out.device.index == 1
    # Compare to reference on cuda:1
    ref = _nearest_centroid_reference(y, c)
    assert torch.equal(out, ref)


@multi_gpu_only
def test_interleaved_device_calls():
    """Alternating calls between cuda:0 and cuda:1 must not cross-contaminate."""
    c0 = torch.linspace(-1.0, 1.0, 8, device="cuda:0")
    c1 = torch.linspace(-1.0, 1.0, 8, device="cuda:1")
    for _ in range(4):
        y0 = torch.randn(64, device="cuda:0")
        y1 = torch.randn(64, device="cuda:1")
        out0 = mse_nearest_centroid(y0, c0)
        out1 = mse_nearest_centroid(y1, c1)
        assert out0.device.index == 0
        assert out1.device.index == 1
        assert torch.equal(out0, _nearest_centroid_reference(y0, c0))
        assert torch.equal(out1, _nearest_centroid_reference(y1, c1))


# ---------------------------------------------------------------------------
# TurboQuantMSE end-to-end: new vs old path must give identical quantize/dequantize.
# ---------------------------------------------------------------------------

class TestTurboQuantMSEEquivalence:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_roundtrip_on_cpu(self, bits):
        torch.manual_seed(0)
        D = 64
        device = torch.device("cpu")
        tq = TurboQuantMSE(D, bits, device=device, dtype=torch.float32, seed=42)
        x = torch.randn(50, D)

        q = tq.quantize(x)
        x_hat = tq.dequantize(q)

        # Reconstruction should preserve norms and reduce MSE monotonically with bits.
        assert x_hat.shape == x.shape
        mse = (x - x_hat).pow(2).mean().item()
        assert mse < 5.0   # sanity: something was reconstructed

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_triton_and_reference_agree_end_to_end(self, bits):
        """Quantize using both paths, verify identical packed indices."""
        torch.manual_seed(0)
        D = 64
        device = torch.device("cpu")
        tq = TurboQuantMSE(D, bits, device=device, dtype=torch.float32, seed=42)
        x = torch.randn(20, D)

        # Manually run the old reference-pipeline for comparison
        x_float = x.float()
        norms = x_float.norm(dim=-1, keepdim=False)
        x_unit = x_float / (norms.unsqueeze(-1) + 1e-10)
        y = x_unit @ tq.pi.T
        ref_indices = _nearest_centroid_reference(y, tq.centroids)
        ref_packed = _pack_indices(ref_indices, bits)

        # New path (via quantize, which now uses mse_nearest_centroid)
        q = tq.quantize(x)

        assert torch.equal(q.indices, ref_packed)
        assert torch.equal(q.norms, norms)


@cuda_only
@pytest.mark.parametrize("bits", [2, 4, 8])
def test_turboquant_mse_cuda_matches_manually_built_cpu(bits):
    """Build a CUDA TurboQuantMSE, then rebuild the SAME pi/centroids on CPU
    (transferring the tensors, not re-sampling from CPU's generator) and
    verify Triton quantize output matches the CPU reference pipeline
    bit-for-bit.

    This is the correct equivalence test: ``random_rotation`` uses a
    device-specific generator, so seed=42 on CPU ≠ seed=42 on CUDA. The
    meaningful invariant is "same pi, same centroids → same indices",
    not "same seed → same pi".
    """
    torch.manual_seed(0)
    D = 64
    tq_cuda = TurboQuantMSE(D, bits, device=torch.device("cuda"), dtype=torch.float32, seed=42)
    x = torch.randn(20, D, device="cuda")

    q_cuda = tq_cuda.quantize(x)

    # Manually replay the pipeline on CPU with the same pi/centroids.
    pi_cpu = tq_cuda.pi.cpu()
    centroids_cpu = tq_cuda.centroids.cpu()
    x_cpu = x.cpu().float()
    norms_cpu = x_cpu.norm(dim=-1, keepdim=False)
    y_cpu = (x_cpu / (norms_cpu.unsqueeze(-1) + 1e-10)) @ pi_cpu.T
    ref_idx = _nearest_centroid_reference(y_cpu, centroids_cpu)
    ref_packed = _pack_indices(ref_idx, bits)

    assert torch.equal(q_cuda.indices.cpu(), ref_packed)
    assert torch.allclose(q_cuda.norms.cpu(), norms_cpu, atol=1e-6)
