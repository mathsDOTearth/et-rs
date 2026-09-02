// Stages the linker script into OUT_DIR and passes it to rustc as an absolute
// path. Using OUT_DIR (rather than CARGO_MANIFEST_DIR directly) makes the link
// argument a build artifact: it is always present regardless of the working
// directory from which cargo is invoked, and `cargo clean` removes it with
// the rest of the target directory. The source link.ld is unchanged.
use std::path::Path;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set by cargo");
    let out_dir = std::env::var("OUT_DIR")
        .expect("OUT_DIR not set by cargo");

    let src = Path::new(&manifest_dir).join("link.ld");
    let dst = Path::new(&out_dir).join("link.ld");
    std::fs::copy(&src, &dst)
        .unwrap_or_else(|e| panic!("failed to copy link.ld to OUT_DIR: {e}"));

    println!("cargo:rustc-link-arg=-T{}", dst.display());
    // Re-run if the source linker script changes; the copy is regenerated automatically.
    println!("cargo:rerun-if-changed=link.ld");
}
