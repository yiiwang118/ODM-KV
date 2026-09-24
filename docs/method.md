# Method notes

ODM-KV assigns one precision to each `(layer, KV head, token)` entry. Keys and
values use the same nominal bit level. Allocation is independent for each
request and layer, and shared across that layer's heads and tokens.

## Attention-output sensitivity

For independent K/V reconstruction errors, the first-order approximation
separates entry sensitivity from quantizer distortion:

$$
\mathbb{E}\|\Delta o\|^2\approx\sum_i s_i\varepsilon(b_i),\qquad
s_i=\hat p_i^2\left[\|v_i\|^2+\frac{\|k_i\|^2}{d}\|v_i-\bar o\|^2\right].
$$

This is a sensitivity surrogate under the noise-model assumptions, rather than
an exact prediction of task loss or arbitrary multi-token eviction error.
The implementation uses `(p + epsilon) ** 2`, with `epsilon=0.01`, to smooth the
attention factor. This parameter is distinct from the quantizer error curve.

Prefill uses the mean query excluding sink positions, rotated by average RoPE
over the next 512 positions. Attention is normalized separately for every query
head, then averaged over the GQA group. The estimated output is its weighted
value sum. Scores retain their raw magnitudes; a rank-preserving transformation
need not preserve the allocation.

The reference path projects the prefill hidden states to obtain queries. The
native path reuses the attention operation's Q/K/V, inverts RoPE to recover the
query mean, and uses fused reductions. Floating-point arithmetic can differ
between these paths; tests check the score and allocation components directly.

## Joint allocation

The candidate set is `{0, 2, 3, 4, 8, 16}`. The cost curve uses calibrated
TurboQuant-MSE reconstruction errors at quantized levels, a separate eviction
endpoint of 0.5, and zero at full precision.

For fixed `lambda`, minimize `score * cost[bits] + lambda * bits` independently
for each entry. Binary search exploits the non-increasing allocated budget as
the bit price rises. A score-ordered repair closes the discrete budget gap.
The repair is heuristic and can leave a small rounding mismatch; it is not an
exact integer-program solution or a strict physical-byte capacity constraint.

Calibration uses generated vectors and is cached by its parameters. It does
not use benchmark labels. No fixed eviction or precision fractions are needed.

## Cache lifecycle

1. Prefill computes an uncompressed layer output, then scores and compresses
   that layer's cache.
2. Four sink tokens remain exact. The last 128 prefill positions are forced to
   full precision after the allocation.
3. Decode appends new tokens to an exact buffer. Once it fills, the current
   query supplies attention over the buffer and the allocator solves a fresh
   nonzero-bit budget.
4. Generated tokens are never evicted. A target below the lowest nonzero
   precision is clamped to that precision for decode, so 1-bit and 2-bit targets
   both flush generated tokens at 2 bits.

The reference evaluator may append a multi-token question after compressing a
shared context. That question suffix stays exact. Native generation takes the
complete prompt in prefill and supports single-token decode; the two prompt
protocols should not be conflated when comparing generated answers.

## Quantization and storage

K and V use separate seeded random rotations and Lloyd–Max scalar codebooks.
Vector norms restore their original scales. The native shared packer stores
3-bit codes in a dense three-bit stream. The reference quantizer uses padded
4-bit containers for nominal 3-bit indices, so its allocation labels are not a
measurement of physical storage.

Optional outlier-channel separation remains available in the reference path.
It is not part of the native fused backend's supported configuration.
