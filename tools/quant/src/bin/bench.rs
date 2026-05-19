// CodeRankEmbed (NomicBert) candle/Metal precision bench. Modes:
//
//   bench run   <model_dir> <csn.json> <N>           speed + MRR@10 + R@1
//   bench equiv <f32_dir> <f16_dir> <csn.json> <N>   cosine f16~f32 + parity
//
// `run` under `/usr/bin/time -l` isolates peak RSS per dtype. `equiv`
// proves a cast is safe: cosine(f16,f32), top-1 retrieval agreement,
// and CodeSearchNet MRR@10 / Recall@1 deltas. CSN JSON is a prefetched
// array of {"doc","code"} (see tools/quant/README.md).
use candle_core::Device;
use quant::{dot, load_csn, retrieval, Enc};
use std::time::Instant;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let dev = Device::new_metal(0).expect("metal");
    match a.get(1).map(String::as_str) {
        Some("run") => {
            let (dir, csn, n) = (&a[2], &a[3], a[4].parse::<usize>().unwrap());
            let (q, d) = load_csn(csn, n);
            let e = Enc::load(dir, &dev);
            let qv = e.embed(&q);
            let t = Instant::now();
            let dv = e.embed(&d);
            let secs = t.elapsed().as_secs_f64();
            let (mrr, r1) = retrieval(&qv, &dv);
            println!(
                "dtype={:?} N={} doc_embed={:.2}s {:.1} docs/s  MRR@10={:.4} R@1={:.4}",
                e.dtype,
                d.len(),
                secs,
                d.len() as f64 / secs,
                mrr,
                r1
            );
        }
        Some("equiv") => {
            let (f32d, f16d, csn, n) = (&a[2], &a[3], &a[4], a[5].parse::<usize>().unwrap());
            let (q, d) = load_csn(csn, n);
            let e32 = Enc::load(f32d, &dev);
            let e16 = Enc::load(f16d, &dev);
            let d32 = e32.embed(&d);
            let d16 = e16.embed(&d);
            let q32 = e32.embed(&q);
            let q16 = e16.embed(&q);
            let nn = d.len();
            let mut sum = 0.0f64;
            let mut min = 1.0f32;
            for (x, y) in d32.iter().zip(&d16) {
                let c = dot(x, y);
                sum += c as f64;
                min = min.min(c);
            }
            let argmax = |qv: &[Vec<f32>], dv: &[Vec<f32>]| -> Vec<usize> {
                (0..qv.len())
                    .map(|i| {
                        (0..dv.len())
                            .max_by(|&x, &y| {
                                dot(&qv[i], &dv[x]).partial_cmp(&dot(&qv[i], &dv[y])).unwrap()
                            })
                            .unwrap()
                    })
                    .collect()
            };
            let a32 = argmax(&q32, &d32);
            let a16 = argmax(&q16, &d16);
            let agree = a32.iter().zip(&a16).filter(|(x, y)| x == y).count() as f64 / nn as f64;
            let (m32, r32) = retrieval(&q32, &d32);
            let (m16, r16) = retrieval(&q16, &d16);
            println!("docs N={nn}");
            println!("cosine(f16,f32): mean={:.6} min={:.6}", sum / nn as f64, min);
            println!("top-1 retrieval agreement f16==f32: {agree:.4}");
            println!("MRR@10  f32={m32:.4}  f16={m16:.4}  Δ={:+.4}", m16 - m32);
            println!("R@1     f32={r32:.4}  f16={r16:.4}  Δ={:+.4}", r16 - r32);
        }
        _ => eprintln!("usage: run <dir> <csn> <N> | equiv <f32> <f16> <csn> <N>"),
    }
}
