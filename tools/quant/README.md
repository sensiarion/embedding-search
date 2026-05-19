# quant — model precision cast + safety bench

Standalone, **Apple-Silicon only** dev tooling. Not a workspace member
(the root `[workspace]` excludes it) so it never enters the main build
or CI. Build/run from this directory.

Used to produce + validate the f16 `nomic-ai/CodeRankEmbed` export
(`sensiarion/CodeRankEmbed-f16`) the main crate loads on the Metal GPU.
Reusable for the same workflow on other models.

## `cast` — safetensors precision cast (architecture-agnostic)

Pure dtype cast on CPU; preserves tensor names/shapes so any loader
reads the output unchanged. Works for **any** safetensors model.

```sh
cargo run --release --bin cast -- in/model.safetensors out/model.safetensors f16
# target: f16 (default) | bf16 | f32
```

Then assemble the export dir = the cast `model.safetensors` +
`config.json` + `tokenizer.json` copied from the base repo, and upload:

```sh
hf upload <user>/<repo> ./out . --repo-type model
```

## `bench` — speed + f16≡f32 safety (NomicBert / CodeRankEmbed family)

`Enc::load` uses `candle_transformers::nomic_bert`; a NomicBert-arch
model works as-is. A different architecture needs its own model in
`src/lib.rs::Enc::load` — the CSN loader and cosine/MRR metrics are
generic.

Prefetch a CodeSearchNet sample (no Python deps beyond stdlib):

```sh
python3 - <<'EOF'
import urllib.request, json
base="https://datasets-server.huggingface.co/rows?dataset=code-search-net/code_search_net&config=python&split=test"
out=[]
for off in (0,100,200):
    d=json.load(urllib.request.urlopen(f"{base}&offset={off}&length=100",timeout=60))
    for r in d["rows"]:
        row=r["row"]; doc=(row.get("func_documentation_string") or "").strip(); code=(row.get("func_code_string") or "").strip()
        if doc and code: out.append({"doc":doc,"code":code})
json.dump(out, open("/tmp/csn.json","w")); print("rows", len(out))
EOF
```

```sh
# precision equivalence (proves a cast is safe to ship)
cargo run --release --bin bench -- equiv <f32_dir> <f16_dir> /tmp/csn.json 300

# per-dtype speed + RSS (run each under /usr/bin/time -l)
/usr/bin/time -l cargo run --release --bin bench -- run <model_dir> /tmp/csn.json 300
```

`equiv` reports `cosine(f16,f32)` mean/min, top-1 retrieval agreement,
and CodeSearchNet MRR@10 / Recall@1 deltas. The absolute MRR is high
vs the published full-CSN number because the sample is a small
distractor pool — it is a **parity** proxy (f16 vs f32), not a CSN
reproduction.

### Result for `sensiarion/CodeRankEmbed-f16` (N=300, CSN python)

| dtype | doc embed | docs/s | peak RSS | MRR@10 | R@1 |
|-------|-----------|--------|----------|--------|-----|
| f32   | 30.68s    | 9.8    | 1116 MB  | 0.9573 | 0.9367 |
| f16   | 28.20s    | 10.6   | 570 MB   | 0.9573 | 0.9367 |

`cosine(f16,f32)` mean **0.999998**, min 0.999996; top-1 agreement
**1.0000**; MRR/R@1 deltas **0.0000**. f16 is numerically a no-op for
retrieval at ~half the RAM.

## The Metal link gotcha

`build.rs` force-links `Metal/Foundation/QuartzCore/CoreGraphics`.
Without it candle's Metal init panics (`swap_remove index (is 0)
should be < len (is 0)`) — objc2-metal declares the externs but does
not link the framework, so `MTLCreateSystemDefaultDevice` returns
NULL. Same fix as the main crate's `build.rs`.
