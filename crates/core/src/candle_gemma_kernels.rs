//! T4 — fused `(x_in + sub)` then Gemma RmsNorm Metal kernel (Opt 4
//! Phase A). Dual-output: produces both the residual-summed value
//! `x = x_in + sub` and the normed output `y = (1 + w) · x · rsqrt(mean(x²) + eps)`
//! in a single MSL dispatch. Gemma3 normalizes BEFORE the residual
//! add, so the residual-summed value is needed twice per layer — a
//! single-output fuse would force a recompute that kills the win.
//!
//! See `docs/OPT4-METAL-KERNELS-PLAN.md` Phase A for the gate
//! criteria.
//!
//! Apple-Silicon only; gated through `candle_backend` from lib.rs.
//! Some helpers (`pipeline_for`, internal layout helpers) only have
//! call sites once `Layer::forward` is wired below — silence the
//! intermediate-build dead-code lint for the whole module.
#![allow(dead_code)]

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CustomOp3, DType, Layout, MetalStorage, Shape, Tensor};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

const MSL_SRC: &str = include_str!("candle_gemma_kernels.metal");
const FN_NAME: &str = "fused_add_rmsnorm_gemma_f32";
/// Threads per row threadgroup. Must be power of 2 — the kernel's
/// pairwise reduction assumes it. 256 fits well under Metal's
/// 1024-thread max with low launch latency.
const TG_SIZE: usize = 256;

/// Pipeline state cache keyed by raw `MetalDevice` pointer.
/// `MTLLibrary` compile is one-time per device and reusing
/// `ComputePipelineState` across forward passes is the standard
/// candle pattern.
fn pipeline_for(
    device: &candle_core::MetalDevice,
) -> Result<candle_metal_kernels::metal::ComputePipeline, String> {
    type Cache = HashMap<usize, candle_metal_kernels::metal::ComputePipeline>;
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
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

/// CustomOp3 fusing `(x_in + sub)` then Gemma RmsNorm, dual output.
///
/// All three inputs must be f32 + contiguous + matching shape (apart
/// from `weight` which is the per-feature vector of size `h = last
/// dim`). Output shape is `[2, ..input_shape]`:
///
///   * `out.i(0)` = `x_in + sub` (the residual sum)
///   * `out.i(1)` = `(1 + weight) · (x_in + sub) · rsqrt(mean(·²) + eps)`
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

fn output_shape_for(input_dims: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(input_dims.len() + 1);
    out.push(2);
    out.extend_from_slice(input_dims);
    out
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
        // Packed: [resid_sum (N*h) | y (N*h)].
        let mut out = vec![0f32; 2 * n * h];
        let y_base = n * h;
        for row in 0..n {
            let mut sum_sq = 0f64;
            for j in 0..h {
                let v = x[off_x + row * h + j] + sub[off_s + row * h + j];
                out[row * h + j] = v;
                sum_sq += (v as f64) * (v as f64);
            }
            let inv = (1.0 / (sum_sq / h as f64 + self.eps as f64).sqrt()) as f32;
            for j in 0..h {
                let v = out[row * h + j];
                out[y_base + row * h + j] = (1.0 + w[off_w + j]) * v * inv;
            }
        }
        Ok((
            CpuStorage::F32(out),
            Shape::from(output_shape_for(l1.shape().dims())),
        ))
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
        // Packed [2 * N * h] output buffer.
        let out_buf = device.new_buffer(2 * n * h, DType::F32, "fused_add_rmsnorm_gemma_out")?;
        let encoder = device.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let dt = DType::F32.size_in_bytes();
        encoder.set_buffer(0, Some(s1.buffer()), l1.start_offset() * dt);
        encoder.set_buffer(1, Some(s2.buffer()), l2.start_offset() * dt);
        encoder.set_buffer(2, Some(s3.buffer()), l3.start_offset() * dt);
        encoder.set_buffer(3, Some(&out_buf), 0);
        let h_u32: u32 = h as u32;
        encoder.set_bytes_directly(4, std::mem::size_of::<u32>(), (&h_u32 as *const u32).cast());
        let n_u32: u32 = n as u32;
        encoder.set_bytes_directly(5, std::mem::size_of::<u32>(), (&n_u32 as *const u32).cast());
        let eps = self.eps;
        encoder.set_bytes_directly(6, std::mem::size_of::<f32>(), (&eps as *const f32).cast());
        encoder.set_threadgroup_memory_length(0, TG_SIZE * std::mem::size_of::<f32>());
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
        let storage = MetalStorage::new(out_buf, device.clone(), 2 * n * h, DType::F32);
        Ok((storage, Shape::from(output_shape_for(l1.shape().dims()))))
    }
}

/// Apply the dual-output fused op to a residual-add + RmsNorm pair.
/// Inputs: f32 + contiguous + matching shape; `weight` is the
/// per-feature vector of length `hidden_size`. Returns a tensor of
/// shape `[2, ..input_shape]`:
///
///   * `result.i(0)` = `x_in + sub` (residual sum)
///   * `result.i(1)` = Gemma RmsNorm of the residual sum
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
    use candle_core::{Device, IndexOp};

    fn ref_impl(x_in: &[f32], sub: &[f32], w: &[f32], h: usize, eps: f32) -> (Vec<f32>, Vec<f32>) {
        let n = x_in.len() / h;
        let mut resid = vec![0f32; n * h];
        let mut y = vec![0f32; n * h];
        for r in 0..n {
            let mut sq = 0f64;
            for j in 0..h {
                let v = x_in[r * h + j] + sub[r * h + j];
                resid[r * h + j] = v;
                sq += (v as f64) * (v as f64);
            }
            let inv = (1.0 / (sq / h as f64 + eps as f64).sqrt()) as f32;
            for j in 0..h {
                y[r * h + j] = (1.0 + w[j]) * resid[r * h + j] * inv;
            }
        }
        (resid, y)
    }

    fn det_data(n: usize, h: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let x: Vec<f32> = (0..n * h).map(|i| ((i as f32) * 0.013).sin()).collect();
        let s: Vec<f32> = (0..n * h).map(|i| ((i as f32) * 0.027).cos()).collect();
        let w: Vec<f32> = (0..h).map(|i| ((i as f32) * 0.05).sin() * 0.3).collect();
        (x, s, w)
    }

    fn validate(got_resid: &[f32], got_y: &[f32], want_resid: &[f32], want_y: &[f32], tol: f32) {
        let mut max_resid = 0f32;
        let mut max_y = 0f32;
        for (a, b) in got_resid.iter().zip(want_resid.iter()) {
            max_resid = max_resid.max((a - b).abs());
        }
        for (a, b) in got_y.iter().zip(want_y.iter()) {
            max_y = max_y.max((a - b).abs());
        }
        assert!(max_resid < tol, "residual err {max_resid}");
        assert!(max_y < tol, "y err {max_y}");
    }

    #[test]
    fn cpu_matches_reference() -> candle_core::Result<()> {
        let (h, n) = (768usize, 4usize);
        let (x, s, w) = det_data(n, h);
        let dev = Device::Cpu;
        let xt = Tensor::from_slice(&x, (n, h), &dev)?;
        let st = Tensor::from_slice(&s, (n, h), &dev)?;
        let wt = Tensor::from_slice(&w, (h,), &dev)?;
        let packed = fused_add_rmsnorm_gemma(&xt, &st, &wt, 1e-6)?;
        assert_eq!(packed.shape().dims(), &[2, n, h]);
        let resid: Vec<f32> = packed
            .i(0)?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let y: Vec<f32> = packed
            .i(1)?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let (want_r, want_y) = ref_impl(&x, &s, &w, h, 1e-6);
        validate(&resid, &y, &want_r, &want_y, 1e-5);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_matches_cpu() -> candle_core::Result<()> {
        let dev = Device::new_metal(0)?;
        let (h, n) = (768usize, 8usize);
        let (x, s, w) = det_data(n, h);
        let xt = Tensor::from_slice(&x, (n, h), &dev)?;
        let st = Tensor::from_slice(&s, (n, h), &dev)?;
        let wt = Tensor::from_slice(&w, (h,), &dev)?;
        let packed = fused_add_rmsnorm_gemma(&xt, &st, &wt, 1e-6)?;
        assert_eq!(packed.shape().dims(), &[2, n, h]);
        let resid: Vec<f32> = packed
            .i(0)?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let y: Vec<f32> = packed
            .i(1)?
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();
        let (want_r, want_y) = ref_impl(&x, &s, &w, h, 1e-6);
        validate(&resid, &y, &want_r, &want_y, 5e-5);
        Ok(())
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn metal_higher_rank() -> candle_core::Result<()> {
        // Backbone calls this with `[b, t, h]`. Output should be
        // `[2, b, t, h]`.
        let dev = Device::new_metal(0)?;
        let (b, t, h) = (2usize, 5usize, 768usize);
        let n = b * t;
        let (x, s, w) = det_data(n, h);
        let xt = Tensor::from_slice(&x, (b, t, h), &dev)?;
        let st = Tensor::from_slice(&s, (b, t, h), &dev)?;
        let wt = Tensor::from_slice(&w, (h,), &dev)?;
        let packed = fused_add_rmsnorm_gemma(&xt, &st, &wt, 1e-6)?;
        assert_eq!(packed.shape().dims(), &[2, b, t, h]);
        let resid: Vec<f32> = packed.i(0)?.flatten_all()?.to_vec1::<f32>()?;
        let y: Vec<f32> = packed.i(1)?.flatten_all()?.to_vec1::<f32>()?;
        let (want_r, want_y) = ref_impl(&x, &s, &w, h, 1e-6);
        validate(&resid, &y, &want_r, &want_y, 5e-5);
        Ok(())
    }
}
