// candle 0.10 uses objc2-metal, which declares the Metal externs but
// does NOT force-link the framework. `MTLCreateSystemDefaultDevice`
// then resolves yet returns NULL at runtime, so candle's metal device
// init finds no GPU and panics. Linking Metal (and the companion
// frameworks the working `metal` crate links) makes the call return
// the real device. Apple-Silicon only — candle is a dependency there.
fn main() {
    // Single source of the candle-backend gate so the 3-clause predicate
    // isn't copy-pasted across the crate. Enabled iff the candle deps
    // are actually compiled in: Apple Silicon, and not the bench-stub
    // (no-model) build.
    println!("cargo:rustc-check-cfg=cfg(candle_backend)");
    let target = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let bench_stub = std::env::var_os("CARGO_FEATURE_BENCH_STUB").is_some();
    if target == "macos" && arch == "aarch64" {
        for fw in ["Metal", "Foundation", "QuartzCore", "CoreGraphics"] {
            println!("cargo:rustc-link-lib=framework={fw}");
        }
        if !bench_stub {
            println!("cargo:rustc-cfg=candle_backend");
        }
    }
}
