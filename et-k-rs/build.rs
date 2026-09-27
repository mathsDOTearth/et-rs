// Stages the linker script into OUT_DIR and passes it to rustc as an absolute
// path. Using OUT_DIR (rather than CARGO_MANIFEST_DIR directly) makes the link
// argument a build artifact: it is always present regardless of the working
// directory from which cargo is invoked, and `cargo clean` removes it with
// the rest of the target directory. The source link.ld is unchanged.
//
// The linker script is only applied when targeting riscv64; on any other target
// (host tests, IDE type-checks) it is skipped so the host linker is not given
// a RISC-V memory layout that would reject a normal executable.
//
// The script also emits `cfg(et_fp_registers)` when the target has the RISC-V
// F extension. `cfg(target_feature = "f")` cannot serve: the RISC-V `f` and `d`
// target features are unstable, and stable rustc never exposes them to `cfg`,
// even where they are enabled.
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=link.ld");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo::rustc-check-cfg=cfg(et_fp_registers)");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    if arch != "riscv64" {
        return;
    }

    if target_has_fp_registers() {
        println!("cargo:rustc-cfg=et_fp_registers");
    }

    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set by cargo");
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set by cargo");

    let src = Path::new(&manifest_dir).join("link.ld");
    let dst = Path::new(&out_dir).join("link.ld");
    std::fs::copy(&src, &dst).unwrap_or_else(|e| panic!("failed to copy link.ld to OUT_DIR: {e}"));

    println!("cargo:rustc-link-arg=-T{}", dst.display());
}

/// Whether the RISC-V target includes the F extension: either the triple's ISA
/// string names it (`g` implies `f` and `d`), or `-C target-feature` enables
/// `+f` or `+d`. The crate's own configuration uses the latter
/// (`riscv64imac` with `+f`), since the Minion has no D extension.
fn target_has_fp_registers() -> bool {
    let target = std::env::var("TARGET").unwrap_or_default();
    let isa = target
        .split('-')
        .next()
        .and_then(|arch| arch.strip_prefix("riscv64"))
        .unwrap_or("");
    let isa_has_f = isa.contains(['g', 'f', 'd']);

    let rustflags = std::env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default();
    let flags_enable_f = rustflags.split('\x1f').any(|flag| {
        flag.contains("target-feature")
            && flag
                .split([',', '='])
                .any(|feature| feature == "+f" || feature == "+d")
    }) || rustflags
        .split('\x1f')
        .collect::<Vec<_>>()
        .windows(2)
        .any(|pair| {
            pair[0] == "-C"
                && pair[1].starts_with("target-feature")
                && pair[1]
                    .split([',', '='])
                    .any(|feature| feature == "+f" || feature == "+d")
        });

    isa_has_f || flags_enable_f
}
