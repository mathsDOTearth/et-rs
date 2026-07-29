# Regenerating bindings

The FFI bindings to the SDK headers are **vendored**: `et-rs/src/bindings_ops.rs`
and `et-rs/src/bindings_trace.rs` are committed to the repository. A default
`cargo build` uses them and needs neither the SDK nor `bindgen`, which is what
lets the crate build on docs.rs and on machines without the SDK.

## When to regenerate

Only when the SDK headers change. Regeneration is gated behind a feature so it
never runs by accident:

```bash
cargo build --features regenerate-bindings
```

This requires the SDK headers under `/opt/et` (or `ET_SDK_PREFIX`) and `libclang`.
The build script runs `bindgen` over the two wrapper headers and rewrites the
committed `bindings_*.rs`, which you then commit.

## Why some things are hand-written

`bindgen` 0.72 miscompiles the SDK's `packed`+`aligned` message structs (a nested
`packed`+`aligned` layout triggers `error[E0588]`). Rather than fight it, the
crate:

- **hand-writes the device-ops message structs** in `et_soc1::proto` as plain
  `#[repr(C)]` types with compile-time size assertions, and blocklists the
  problematic op-stats struct from generation;
- **generates two separate trace wrappers**, because the SDK defines
  `enum trace_buffer_type` twice with conflicting values
  (`et_ioctl.h` versus `et-trace/layout.h`); keeping them in separate modules
  avoids the clash;
- **computes the ioctl request codes** with a `const fn` mirroring `_IOC`, guarded
  by the golden test `ioctl::tests::request_codes_match_c_header` so they cannot
  silently diverge from the header.

The generated bindings are therefore deliberately a subset: the parts `bindgen`
handles cleanly (enums, constants, simple structs), with the awkward parts owned
by hand-written, tested code in `proto` and `ioctl`.

## What is not in the repository

The SDK itself (headers, libraries, firmware ELFs) and the vendor `et-testdrive`
sources are **not** checked in. They are needed only for the emulator backend, for
regenerating bindings, and at run time; the committed bindings and hand-written
protocol code are enough for a default build.
