// Stages the linker script into OUT_DIR and passes it to rustc as an absolute
// path. Using OUT_DIR (rather than CARGO_MANIFEST_DIR directly) makes the link
// argument a build artifact: it is always present regardless of the working
// directory from which cargo is invoked, and `cargo clean` removes it with
// the rest of the target directory. The source link.ld is unchanged.
//
// The linker script is only applied when targeting riscv64; on any other target
// (host tests, IDE type-checks) it is skipped so the host linker is not given
// a RISC-V memory layout that would reject a normal executable.
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=link.ld");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch != "riscv64" {
        return;
    }

    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set by cargo");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set by cargo");

    let src = Path::new(&manifest_dir).join("link.ld");
    let dst = Path::new(&out_dir).join("link.ld");
    std::fs::copy(&src, &dst).unwrap_or_else(|e| panic!("failed to copy link.ld to OUT_DIR: {e}"));

    println!("cargo:rustc-link-arg=-T{}", dst.display());
}
