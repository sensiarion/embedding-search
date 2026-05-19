// Architecture-agnostic safetensors precision cast (CPU). Preserves
// tensor names/shapes so any loader reads the output unchanged; only
// float tensors are converted, integer buffers pass through.
//
//   cargo run --release --bin cast -- <in.safetensors> <out.safetensors> [f16|bf16|f32]
//
// Default target: f16. Reusable for any safetensors model.
use candle_core::safetensors::{save, MmapedSafetensors};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;

fn parse_dtype(s: &str) -> DType {
    match s {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        "f32" => DType::F32,
        other => panic!("unsupported target dtype {other:?} (f16|bf16|f32)"),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let src = args
        .next()
        .expect("usage: cast <in.safetensors> <out.safetensors> [f16|bf16|f32]");
    let dst = args
        .next()
        .expect("usage: cast <in.safetensors> <out.safetensors> [f16|bf16|f32]");
    let target = args.next().as_deref().map(parse_dtype).unwrap_or(DType::F16);
    let dev = Device::Cpu;

    // SAFETY: immutable input file, not mutated for the mmap's life.
    let st = unsafe { MmapedSafetensors::new(&src) }?;
    let names: Vec<String> = st.tensors().into_iter().map(|(n, _)| n).collect();
    println!("{} tensors -> {target:?}", names.len());

    let mut out: HashMap<String, Tensor> = HashMap::with_capacity(names.len());
    for n in &names {
        let t = st.load(n, &dev)?;
        let t = match t.dtype() {
            DType::F32 | DType::F64 | DType::BF16 | DType::F16 => t.to_dtype(target)?,
            _ => t,
        };
        out.insert(n.clone(), t);
    }
    save(&out, &dst)?;
    println!("wrote {dst}");
    Ok(())
}
