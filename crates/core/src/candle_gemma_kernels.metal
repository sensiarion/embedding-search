// Fused residual_add + RmsNorm(Gemma), dual-output variant for the
// EmbeddingGemma candle Metal backbone.
//
// Per-row math (one row = one (b, t) position; row vector of h
// elements):
//
//     x_i      = x_in_i + sub_i                         (residual add)
//     mean_sq  = (Σ x_i²) / h                           (reduction)
//     inv_rms  = 1 / sqrt(mean_sq + eps)
//     y_i      = (1 + weight_i) * x_i * inv_rms         (Gemma's `1 + w`)
//
// Why dual output: Gemma3 normalizes BEFORE the residual add (the
// residual-summed value `x` is needed both as the norm's input AND
// as the saved residual for the layer's final add). A single-output
// fuse forces recomputing `x`, which kills the bandwidth savings.
//
// Layout: one output buffer of size [2 * N * h], row-major.
//
//     out[0           .. N*h]    = x  (residual_sum)
//     out[N*h         .. 2*N*h]  = y  (normed)
//
// Caller unpacks via `Tensor::i(0)` / `i(1)` after reshaping to
// `[2, ..input_shape]`. One threadgroup per row, cooperative
// reduction in threadgroup memory.

#include <metal_stdlib>
using namespace metal;

kernel void fused_add_rmsnorm_gemma_f32(
    device const float* x_in     [[ buffer(0) ]],
    device const float* sub      [[ buffer(1) ]],
    device const float* weight   [[ buffer(2) ]],
    device       float* out      [[ buffer(3) ]],
    constant uint&      h        [[ buffer(4) ]],
    constant uint&      total_n  [[ buffer(5) ]],
    constant float&     eps      [[ buffer(6) ]],
    threadgroup float*  shared   [[ threadgroup(0) ]],
    uint row     [[ threadgroup_position_in_grid ]],
    uint tid     [[ thread_position_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    const uint base   = row * h;
    const uint y_base = total_n * h + row * h;

    // First pass: write residual sum to its output slot AND
    // accumulate sum-of-squares for the rms reduction. Single load
    // per element of x_in / sub.
    float partial = 0.0f;
    for (uint i = tid; i < h; i += tg_size) {
        float xi = x_in[base + i] + sub[base + i];
        out[base + i] = xi;
        partial += xi * xi;
    }
    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pairwise reduction (`tg_size` must be power of two).
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt(shared[0] / float(h) + eps);

    // Second pass: read residual_sum back (cached in L1), apply
    // `(1 + w) * x * inv_rms`, write y.
    for (uint i = tid; i < h; i += tg_size) {
        float xi = out[base + i];
        float w  = weight[i];
        out[y_base + i] = (1.0f + w) * xi * inv_rms;
    }
}
