//! T4 — fused residual_add + Gemma RmsNorm Metal kernel (Opt 4
//! Phase A). Replaces a `(x + sublayer)` op followed by
//! `GemmaRmsNorm::forward` with a single MSL kernel that loads the
//! pair once and writes the normed output, saving 1 read + 1 write
//! per call. Two such pairs per Gemma3 layer × 24 layers = 48
//! dispatches removed per forward.
//!
//! See `docs/OPT4-METAL-KERNELS-PLAN.md` (Phase A) for the design
//! decision gate.
//!
//! Apple-Silicon only; gated through `candle_backend` from lib.rs.
//!
//! Wired into `candle_gemma_embed::Layer::forward` in a follow-up
//! step (allow(dead_code) until then so the scaffolding can land
//! and unit-tests against the CPU/Metal reference can run).
#![allow(dead_code)]

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, MetalStorage, Shape, Tensor};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const MSL_SRC: &str = include_str!("candle_gemma_kernels.metal");
const FN_NAME: &str = "fused_add_rmsnorm_gemma_f32";
/// Threads per row threadgroup. Must be a power of 2 (the kernel's
/// pairwise reduction assumes it). 256 fits comfortably under
/// Metal's 1024-thread max with low launch latency.
const TG_SIZE: usize = 256;

/// Pipeline state cache keyed by raw `MetalDevice` pointer. The
/// MTLLibrary compile is one-time per device and reusing
/// `ComputePipelineState` across forward passes is the standard
/// candle pattern. Pointer key is safe — candle clones an Arc and
/// the underlying object lives for the device's lifetime.
fn pipeline_for(
    device: &candle_core::MetalDevice,
) -> Result<candle_metal_kernels::metal::ComputePipeline, String> {
    type Cache = HashMap<usize, candle_metal_kernels::metal::ComputePipeline>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    // Key by MTLDevice pointer. Candle hands us a long-lived &Device;
    // the address uniquely identifies the GPU within the process.
    let raw = device.device();
    let key = (raw as *const _) as *const () as usize;
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache
        .lock()
        .map_err(|e| format!("kernel pipeline cache poisoned: {e}"))?;
    if let Some(p) = guard.get(&key) {
        let cloned: candle_metal_kernels::metal::ComputePipeline = p.clone();
        return Ok(cloned);
    }
    let opts: Option<&objc2_metal::MTLCompileOptions> = None;
    let lib = raw
        .new_library_with_source(MSL_SRC, opts)
        .map_err(|e| format!("MSL compile: {e}"))?;
    let func = lib
        .get_function(FN_NAME, None)
        .map_err(|e| format!("{FN_NAME}: {e}"))?;
    let pipeline = device
        .device()
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| format!("pipeline state: {e}"))?;
    guard.insert(key, pipeline.clone());
    Ok(pipeline)
}

/// CustomOp3 fusing `(x_in + sub)` then Gemma-style RmsNorm:
///
/// ```text
///   x_i = x_in_i + sub_i
///   y_i = (1 + weight_i) * x_i / sqrt(mean(x²) + eps)
/// ```
///
/// All three inputs must be f32 + contiguous + same shape (apart
/// from `weight` which is the per-feature broadcast vector of size
/// `h = last dim`).
pub struct FusedAddRmsNormGemma {
    pub eps: f32,
}

fn flat_n_h(
    x_l: &Layout,
    sub_l: &Layout,
    w_l: &Layout,
) -> Result<(usize, usize), candle_core::Error> {
    if x_l.shape().dims() != sub_l.shape().dims() {
        return Err(candle_core::Error::Msg(format!(
            "fused_add_rmsnorm: shape mismatch x={:?} sub={:?}",
            x_l.shape().dims(),
            sub_l.shape().dims(),
        )));
    }
    let dims = x_l.shape().dims();
    let h = *dims
        .last()
        .ok_or_else(|| candle_core::Error::Msg("fused_add_rmsnorm: rank 0 tensor".into()))?;
    if w_l.shape().dims() != [h] {
        return Err(candle_core::Error::Msg(format!(
            "fused_add_rmsnorm: weight must be [{}], got {:?}",
            h,
            w_l.shape().dims()
        )));
    }
    if !x_l.is_contiguous() || !sub_l.is_contiguous() || !w_l.is_contiguous() {
        return Err(candle_core::Error::Msg(
            "fused_add_rmsnorm: inputs must be contiguous".into(),
        ));
    }
    let n: usize = dims[..dims.len() - 1].iter().product::<usize>().max(1);
    Ok((n, h))
}

impl CustomOp3 for FusedAddRmsNormGemma {
    fn name(&self) -> &'static str {
        "fused_add_rmsnorm_gemma"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        let (n, h) = flat_n_h(l1, l2, l3)?;
        let x = match s1 {
            CpuStorage::F32(v) => v.as_slice(),
            _ => {
                return Err(candle_core::Error::Msg(
                    "fused_add_rmsnorm: f32 only".into(),
                ))
            }
        };
        let sub = match s2 {
            CpuStorage::F32(v) => v.as_slice(),
            _ => {
                return Err(candle_core::Error::Msg(
                    "fused_add_rmsnorm: f32 only".into(),
                ))
            }
        };
        let w = match s3 {
            CpuStorage::F32(v) => v.as_slice(),
            _ => {
                return Err(candle_core::Error::Msg(
                    "fused_add_rmsnorm: f32 only".into(),
                ))
            }
        };
        let off_x = l1.start_offset();
        let off_s = l2.start_offset();
        let off_w = l3.start_offset();
        let mut out = vec![0f32; n * h];
        for row in 0..n {
            let mut sum_sq = 0f64;
            for j in 0..h {
                let v = x[off_x + row * h + j] + sub[off_s + row * h + j];
                sum_sq += (v as f64) * (v as f64);
            }
            let inv = (1.0 / (sum_sq / h as f64 + self.eps as f64).sqrt()) as f32;
            for j in 0..h {
                let v = x[off_x + row * h + j] + sub[off_s + row * h + j];
                out[row * h + j] = (1.0 + w[off_w + j]) * v * inv;
            }
        }
        Ok((CpuStorage::F32(out), Shape::from(l1.shape().dims())))
    }

    fn metal_fwd(
        &self,
        s1: &MetalStorage,
        l1: &Layout,
        s2: &MetalStorage,
        l2: &Layout,
        s3: &MetalStorage,
        l3: &Layout,
    ) -> candle_core::Result<(MetalStorage, Shape)> {
        use objc2_metal::MTLResourceUsage;
        let (n, h) = flat_n_h(l1, l2, l3)?;
        if s1.dtype() != DType::F32 || s2.dtype() != DType::F32 || s3.dtype() != DType::F32 {
            return Err(candle_core::Error::Msg(
                "fused_add_rmsnorm: f32 only on Metal".into(),
            ));
        }
        let device = s1.device();
        let pipeline = pipeline_for(device).map_err(candle_core::Error::Msg)?;
        let out_buf = device.new_buffer(n * h, DType::F32, "fused_add_rmsnorm_gemma_out")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let dt = DType::F32.size_in_bytes();
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * dt);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * dt);
        encoder.set_buffer(2, Some(s3.buffer()), l3.start_offset() * dt);
        encoder.set_buffer(3, Some(&out_buf), 0);
        let h_u32: u32 = h as u32;
        encoder.set_bytes_directly(4, std::mem::size_of::<u32>(), (&h_u32 as *const u32).cast());
        let eps = self.eps;
        encoder.set_bytes_directly(5, std::mem::size_of::<f32>(), (&eps as *const f32).cast());
        encoder.set_threadgroup_memory_length(0, TG_SIZE * std::mem::size_of::<f32>());
        // `use_resource` takes `impl Into<&MetalResource>`; candle's
        // `Buffer` directly implements `From<&Buffer> for
        // &MetalResource`, so the raw `&Buffer` (from
        // `MetalStorage::buffer()`) is the right shape — no
        // `as_ref()` to a deeper type.
        encoder.use_resource(s1.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s2.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(s3.buffer(), MTLResourceUsage::Read);
        encoder.use_resource(&*out_buf, MTLResourceUsage::Write);
        let groups = objc2_metal::MTLSize {
            width: n,
            height: 1,
            depth: 1,
        };
        let threads = objc2_metal::MTLSize {
            width: TG_SIZE,
            height: 1,
            depth: 1,
        };
        encoder.dispatch_thread_groups(groups, threads);
        let storage = MetalStorage::new(out_buf, device.clone(), n * h, DType::F32);
        Ok((storage, Shape::from(l1.shape().dims())))
    }
}

/// Apply the fused op to a residual-add + RmsNorm pair. Inputs must
/// be f32 + contiguous + matching shape; `weight` is the per-feature
/// vector of length `hidden_size`.
pub fn fused_add_rmsnorm_gemma(
    x_in: &Tensor,
    sub: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> candle_core::Result<Tensor> {
    x_in.apply_op3(sub, weight, FusedAddRmsNormGemma { eps })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn ref_impl(x_in: &[f32], sub: &[f32], w: &[f32], h: usize, eps: f32) -> Vec<f32> {
        let n = x_in.len() / h;
        let mut out = vec![0f32; n * h];
        for r in 0..n {
            let mut sq = 0f64;
            for j in 0..h {
                let v = x_in[r * h + j] + sub[r * h + j];
                sq += (v as f64) * (v as f64);
            }
            let inv = (1.0 / (sq / h as f64 + eps as f64).sqrt()) as f32;
            for j in 0..h {
                let v = x_in[r * h + j] + sub[r * h + j];
                out[r * h + j] = (1.0 + w[j]) * v * inv;
            }
        }
        out
    }

    fn det_data(n: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..n * h).map(|i| ((i as f32) * 0.013).sin()).collect();
        let s: Vec<f32> = (0..n * h).map(|i| ((i as f32) * 0.027).cos()).collect();
        let w: Vec<f32> = (0..h).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
        (x, s, w)
    }

    #[test]
    fn cpu_matches_reference() -> candle_core::Result<()> {
        let h = 768;
        let n = 4;
        let (x, s, w) = det_data(n, h);
        let dev = Device::Cpu;
        let xt = Tensor::from_slice(&x, (n, h), &dev)?;
        let st = Tensor::from_slice(&s, (n, h), &dev)?;
        let wt = Tensor::from_slice(&w, (h,), &dev)?;
        let got: Vec<f32> = fused_add_rmsnorm_gemma(&xt, &st, &wt, 1e-6)?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let want = ref_impl(&x, &s, &w, h, 1e-6);
        for (a, b) in got.iter().zip(want.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_matches_cpu() -> candle_core::Result<()> {
        let dev = Device::new_metal(0)?;
        let h = 768;
        let n = 8;
        let (x, s, w) = det_data(n, h);
        let xt = Tensor::from_slice(&x, (n, h), &dev)?;
        let st = Tensor::from_slice(&s, (n, h), &dev)?;
        let wt = Tensor::from_slice(&w, (h,), &dev)?;
        let got = fused_add_rmsnorm_gemma(&xt, &st, &wt, 1e-6)?.to_vec2::<f32>()?;
        let want = ref_impl(&x, &s, &w, h, 1e-6);
        let mut max_err = 0f32;
        for r in 0..n {
            for j in 0..h {
                let a = got[r][j];
                let b = want[r * h + j];
                max_err = max_err.max((a - b).abs());
            }
        }
        assert!(max_err < 5e-5, "max err {max_err}");
        Ok(())
    }
}
