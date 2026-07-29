//! Build script.
//!
//! By default this does almost nothing: the FFI bindings are vendored in
//! `src/bindings_ops.rs` / `src/bindings_trace.rs` and compiled directly, so a
//! plain `cargo build` needs neither the Esperanto SDK nor bindgen. The SDK is
//! consulted only for:
//!
//! * `--features regenerate-bindings` -- run bindgen against the SDK headers and
//!   rewrite the committed `src/bindings_*.rs` (maintainers only); and
//! * `--features emu` -- compile and link the C++ software-emulator shim.
//!
//! The SDK location defaults to `/opt/et`, overridable via `ET_SDK_PREFIX`.

fn main() {
    println!("cargo:rerun-if-env-changed=ET_SDK_PREFIX");

    #[cfg(feature = "regenerate-bindings")]
    regenerate_bindings();

    #[cfg(feature = "emu")]
    build_emu_shim();
}

/// SDK install prefix (`/opt/et` unless overridden).
#[cfg(any(feature = "regenerate-bindings", feature = "emu"))]
fn sdk_prefix() -> String {
    std::env::var("ET_SDK_PREFIX").unwrap_or_else(|_| "/opt/et".to_string())
}

/// Regenerate the vendored bindings from the SDK headers, writing them back into
/// the source tree so they can be committed.
///
/// Two independent bindgen invocations are used because `et_ioctl.h` and
/// `esperanto/et-trace/layout.h` each define a distinct `enum trace_buffer_type`;
/// combining them in one translation unit would be an ODR violation in C.
#[cfg(feature = "regenerate-bindings")]
fn regenerate_bindings() {
    use std::path::PathBuf;

    let prefix = sdk_prefix();
    let clang_include = format!("-I{prefix}/include");
    let src =
        PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR")).join("src");

    println!("cargo:rerun-if-changed=wrapper_ops.h");
    println!("cargo:rerun-if-changed=wrapper_trace.h");

    // Common configuration shared between both binding sets. Enum variants are
    // emitted as module-scoped constants: the device-ops enums contain several
    // aliased discriminants (e.g. *_ERROR == *_UNEXPECTED_ERROR == 1) which a
    // Rust `enum` cannot represent, and constants side-step the collision.
    let base = || {
        bindgen::Builder::default()
            .clang_arg(&clang_include)
            .default_enum_style(bindgen::EnumVariation::ModuleConsts)
            .derive_debug(true)
            .derive_default(true)
            .derive_copy(true)
            .layout_tests(true)
            .use_core()
            // Wrap generated unsafe operations in explicit `unsafe` blocks so the
            // output compiles cleanly under the Rust 2024 edition's
            // `unsafe_op_in_unsafe_fn` lint.
            .wrap_unsafe_ops(true)
            .ctypes_prefix("::core::ffi")
            .generate_comments(true)
    };

    // --- Driver uapi + device-ops enumerations ---
    //
    // Only `et_ioctl.h` (the character-device descriptors and driver enums) and
    // `device_ops_api_spec.h` (pure enumerations) are generated here. The
    // device-ops *message* structs in device_apis_message_types.h /
    // device_ops_api_rpc_types.h are declared `packed, aligned(8)`; bindgen 0.72
    // lowers that family to a mix of `#[repr(packed)]` and `#[repr(align(8))]`
    // types that nest illegally (rustc E0588). Because the `packed` attribute is
    // a layout no-op for those structs (their fields are already naturally
    // aligned and ordered by decreasing size), they are transcribed by hand in
    // the `proto` module with compile-time layout assertions instead.
    let ops = base()
        .header("wrapper_ops.h")
        .allowlist_file(".*/et_ioctl.h")
        .allowlist_file(".*/device_ops_api_spec.h")
        .generate()
        .expect("failed to generate device-ops bindings");
    ops.write_to_file(src.join("bindings_ops.rs"))
        .expect("failed to write src/bindings_ops.rs");

    // --- et-trace buffer layout ---
    // The SP operating-point statistics structs (op_value/op_module/op_stats_t)
    // and the compute-resource sample structs are `packed, aligned(8)` and hit
    // the same bindgen E0588 nesting defect. They are not needed to decode the
    // buffer stream (they appear only as opaque custom-event payloads), so they
    // are blocklisted here.
    let trace = base()
        .header("wrapper_trace.h")
        .allowlist_file(".*/et-trace/layout.h")
        .blocklist_type("op_value")
        .blocklist_type("op_module")
        .blocklist_type("op_stats_t")
        .blocklist_type("resource_value")
        .blocklist_type("compute_resources_sample")
        .generate()
        .expect("failed to generate et-trace bindings");
    trace
        .write_to_file(src.join("bindings_trace.rs"))
        .expect("failed to write src/bindings_trace.rs");

    println!(
        "cargo:warning=Regenerated src/bindings_ops.rs and src/bindings_trace.rs from {prefix}/include; commit the changes."
    );
}

/// Configure and build `emu-shim` with CMake, then emit the link directives so
/// the crate links the shared shim and finds it (and the SDK libs) at runtime.
///
/// The C++ shim is compiled with CMake using the SDK's own find_package
/// configuration, which resolves the whole transitive link chain (sw-sysemu,
/// hostUtils, linuxDriver, Boost, glog, lz4); Rust then links the resulting
/// shared object and rpaths both it and the SDK lib directory (for libg3log).
#[cfg(feature = "emu")]
fn build_emu_shim() {
    use std::path::PathBuf;

    let prefix = sdk_prefix();
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR set by cargo"));
    let src = format!("{manifest}/emu-shim");
    let build = out_dir.join("emu-shim-build");

    println!("cargo:rerun-if-changed=emu-shim/et_emu_shim.cpp");
    println!("cargo:rerun-if-changed=emu-shim/et_emu_shim.h");
    println!("cargo:rerun-if-changed=emu-shim/CMakeLists.txt");

    let status = std::process::Command::new("cmake")
        .args([
            "-S",
            &src,
            "-B",
            build.to_str().unwrap(),
            &format!("-DCMAKE_PREFIX_PATH={prefix}"),
            "-DCMAKE_BUILD_TYPE=Release",
            "-Wno-dev",
        ])
        .status()
        .expect("failed to run cmake (is it installed?)");
    assert!(status.success(), "cmake configure of emu-shim failed");

    let status = std::process::Command::new("cmake")
        .args(["--build", build.to_str().unwrap(), "--parallel"])
        .status()
        .expect("failed to run cmake --build");
    assert!(status.success(), "cmake build of emu-shim failed");

    let build_dir = build.to_str().unwrap();
    println!("cargo:rustc-link-search=native={build_dir}");
    println!("cargo:rustc-link-lib=dylib=et_emu_shim");
    // Locate the shim and the SDK shared libraries (e.g. libg3log) at runtime.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{build_dir}");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{prefix}/lib");
}
