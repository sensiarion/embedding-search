// candle 0.10 pulls objc2-metal, which declares the Metal externs but
// does NOT force-link the framework, so `MTLCreateSystemDefaultDevice`
// resolves yet returns NULL → candle's Metal device init panics
// (`swap_remove index (is 0) should be < len (is 0)`). Force-linking
// Metal + companions makes the call return the real device. Identical
// fix to the main crate's build.rs. Apple-Silicon only.
fn main() {
    for fw in ["Metal", "Foundation", "QuartzCore", "CoreGraphics"] {
        println!("cargo:rustc-link-lib=framework={fw}");
    }
}
