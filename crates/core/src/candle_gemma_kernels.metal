// Fused residual_add + RmsNorm(Gemma) kernel for the EmbeddingGemma
// candle Metal backbone.
//
// Per-row math (one row per (b, t) position, h-element hidden vector):
//
//     x_i      = x_in_i + sub_i                         (residual add)
//     mean_sq  = (Σ x_i²) / h                           (reduction)
//     inv_rms  = 1 / sqrt(mean_sq + eps)
//     y_i      = (1 + weight_i) * x_i * inv_rms         (Gemma's `1 + w`)
//
// Layout: tensors flattened as `[N rows × h]`, row-major. One
// threadgroup per row; cooperative reduction in threadgroup memory.

#include <metal_stdlib>
using namespace metal;

kernel void fused_add_rmsnorm_gemma_f32(
    device const float* x_in     [[ buffer(0) ]],
    device const float* sub      [[ buffer(1) ]],
    device const float* weight   [[ buffer(2) ]],
    device       float* y        [[ buffer(3) ]],
    constant uint&      h        [[ buffer(4) ]],
    constant float&     eps      [[ buffer(5) ]],
    threadgroup float*  shared   [[ threadgroup(0) ]],
    uint row     [[ threadgroup_position_in_grid ]],
    uint tid     [[ thread_position_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]]
) {
    const uint base = row * h;

    // Partial sum of squares of (x_in + sub) for this row's slice.
    float partial = 0.0f;
    for (uint i = tid; i < h; i += tg_size) {
        float xi = x_in[base + i] + sub[base + i];
        partial += xi * xi;
    }
    shared[tid] = partial;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pairwise reduction. `tg_size` must be power of two.
    for (uint stride = tg_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float inv_rms = rsqrt(shared[0] / float(h) + eps);

    // Apply (1 + w) * x * inv_rms. Second pass recomputes
    // `x_in + sub` per element — cheap vs the bandwidth of a
    // staged write.
    for (uint i = tid; i < h; i += tg_size) {
        float xi = x_in[base + i] + sub[base + i];
        float w  = weight[i];
        y[base + i] = (1.0f + w) * xi * inv_rms;
    }
}
