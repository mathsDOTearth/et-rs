//! Build script: generates FFI bindings for the ET-SoC-1 driver uapi, the
//! device-ops RPC message ABI, and the et-trace buffer layout, from the headers
//! shipped with the Esperanto SDK (default prefix `/opt/et`).
//!
//! The SDK location is overridable via the `ET_SDK_PREFIX` environment variable
//! so that the crate can be built against an SDK installed elsewhere on the
//! remote hardware host.
//!
//! Two independent bindgen invocations are used because `et_ioctl.h` and
//! `esperanto/et-trace/layout.h` each define a distinct `enum trace_buffer_type`;
//! combining them in one translation unit would be an ODR violation in C.

use std::env;
use std::path::PathBuf;

fn main() {
    let prefix = env::var("ET_SDK_PREFIX").unwrap_or_else(|_| "/opt/et".to_string());
    let include = format!("{prefix}/include");
    let clang_include = format!("-I{include}");

    println!("cargo:rerun-if-env-changed=ET_SDK_PREFIX");
    println!("cargo:rerun-if-changed=wrapper_ops.h");
    println!("cargo:rerun-if-changed=wrapper_trace.h");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR set by cargo"));

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
    ops.write_to_file(out_dir.join("bindings_ops.rs"))
        .expect("failed to write bindings_ops.rs");

    // --- Optional software-emulator FFI backend ---
    // Built only under the `emu` feature. The C++ shim is compiled with CMake
    // using the SDK's own find_package configuration, which resolves the whole
    // transitive link chain (sw-sysemu, hostUtils, linuxDriver, Boost, glog,
    // lz4); Rust then links the resulting shared object and rpaths both it and
    // the SDK lib directory (for libg3log).
    if env::var_os("CARGO_FEATURE_EMU").is_some() {
        build_emu_shim(&prefix, &out_dir);
    }

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
        .write_to_file(out_dir.join("bindings_trace.rs"))
        .expect("failed to write bindings_trace.rs");
}

/// Configure and build `emu-shim` with CMake, then emit the link directives so
/// the crate links the shared shim and finds it (and the SDK libs) at runtime.
fn build_emu_shim(prefix: &str, out_dir: &std::path::Path) {
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo");
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
