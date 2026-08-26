// Passes the linker script to rustc as an absolute path so that kernels link
// correctly regardless of the working directory from which cargo is invoked.
// Without an absolute path, `-Tlink.ld` fails when cargo is run from the
// workspace root rather than from within the et-k-rs package directory.
fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set by cargo");
    println!("cargo:rustc-link-arg=-T{manifest_dir}/link.ld");
    // Re-run only if the linker script itself changes.
    println!("cargo:rerun-if-changed=link.ld");
}
