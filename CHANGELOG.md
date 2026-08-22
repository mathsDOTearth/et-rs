# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0/). The three crates
(`et-abi`, `et-rs`, `et-k-rs`) are released together and share a version.

## [0.3.0] - 2026-08-22

### Added

- **`et-abi`**: tensor-extension ABI additions.
  - `GemmArgs`: 64-byte `#[repr(C)]` launch-argument struct for sGEMM (4 x u64
    for pointers and shire count, 8 x u32/f32 for dimensions, strides, and
    scaling). A compile-time size assertion (`assert!(size_of::<GemmArgs>() == 64)`)
    guards the host/device wire layout.
  - `TENSOR_ALIGN` (64 bytes): required alignment for tensor load/store addresses
    and leading dimensions.
  - `GEMM_TILE_M`, `GEMM_TILE_K`, `GEMM_TILE_N` (16 each): tile dimensions of
    the sGEMM kernel.
  - `MINIONS_PER_SHIRE` (32): number of Minion cores per compute shire.

- **`et-k-rs`**: tensor-extension intrinsics and sGEMM kernel.
  - `et_kernel::tensor`: low-level tensor co-processor intrinsics using standard
    RISC-V `csrrw` instructions (no custom opcodes; `riscv64imac` suffices).
    Provides `tensor_load`, `tensor_load_b`, `tensor_fma32`, `tensor_store`,
    `tensor_wait`, `tensor_error`, `set_tensor_mask`, and `fma32_xs` for
    constructing the xs bit field. CSR addresses verified against PRM Table 9-7
    and confirmed on hardware (`CSR_TENSOR_STORE = 0x87F`, not the erroneous
    `0x83E` originally derived from memory).
  - `et_kernel::simd`: stub module for the ET-SoC-1 packed-single (PS) SIMD
    extension, gated on `cfg(target_feature = "f")`. Full implementations await
    confirmed PS opcode encodings from PRM Chapter 5. The module is marked
    `#[doc(hidden)]` and excluded from published documentation until the asm
    bodies are verified.
  - `sgemm-rs` binary (`src/bin/sgemm.rs`): single-precision GEMM kernel using
    the tensor extension. Supports alpha=1.0, beta=0.0, N a multiple of 16.
    Tile assignment is cyclic across Minion cores; only primary harts
    (`mhartid & 1 == 0`) issue tensor instructions.

- **`et-rs`**: host-side BLAS launcher.
  - `et_soc1::blas::sgemm`: launches `sgemm-rs` to compute C = alpha*A*B + beta*C.
    Validates alignment, dimension, and v0.1 scaling constraints before launch.
  - `et_soc1::blas::alloc_tensor_matrix`: allocates a device buffer with row
    stride padded to `TENSOR_ALIGN` bytes.
  - `et_soc1::blas::GemmError`: typed error enum covering alignment, dimension,
    and unsupported-scaling violations. Re-exported at the crate root as
    `et_soc1::GemmError` so callers do not need to import the `blas` sub-module.
  - `examples/sgemm.rs`: end-to-end demonstration (64x64x64) that uploads
    A and B, launches, downloads C, and verifies spot-checked elements against
    a scalar reference. Verified on hardware: C[3][7] = 168.0000 with zero error.

## [0.2.0] - 2026-07-30

A major reorganisation and feature expansion of the `et-rs` host crate (first
published at 0.1.0), split into a Cargo workspace, and the first release of the
companion crates `et-abi` and `et-k-rs`. All three now share this version.

### Added

- **`et-abi`** (`et_abi`): shared, `no_std`, dependency-free host/device ABI. The
  `DeviceArgs` trait and the `ReduceArgs` launch-argument struct are defined once,
  so the host launcher and the device kernel cannot drift on layout. Also holds
  the shared architectural constants (`CACHE_LINE`, `HARTS_PER_SHIRE`,
  `HARTS_PER_NEIGHBOURHOOD`), so the two sides agree on cache-line stride and
  geometry.
- **`et-rs`** (`et_soc1`): pure-Rust host driver for the ET-SoC-1.
  - Kernel loading (DMA-write of `PT_LOAD` segments), SPMD launch, DMA transfer,
    and a pure-Rust et-trace decoder.
  - Two backends behind a `Transport` trait: `IoctlTransport` (real hardware) and
    `FfiTransport` (SDK software emulator, `emu` feature).
  - Typed device memory: `DeviceBuffer<T>`, `PaddedArray<T>`, and the `DevicePod`
    marker, with `upload` / `download` / `alloc_array` / `alloc_padded` /
    `download_padded`.
  - `Device::topology()` and `Topology`: device-queried compute-shire mask plus
    architectural per-shire geometry, so callers need not hard-code hart counts.
  - Typed SPMD launch: `Device::launch_spmd` / `launch_spmd_traced`.
  - DRAM arena reclamation: `Device::alloc_mark` / `reset_to`; `launch` reuses an
    internal argument scratch region rather than leaking one per call.
  - Decoded launch failures: `Error::KernelLaunch` carrying the symbolic status
    name, the faulting shire mask, and the exception/trace buffer pointers.
- **`et-k-rs`** (`et_kernel`): device-side library for writing compute kernels in
  pure `no_std` Rust, plus demo kernels (`hello-rs`, `reduce-rs`, `spsc-rs`). The
  `Grid` abstraction gives each hart only its disjoint slice and its own
  cache-line-padded output cell, making cross-hart data races unrepresentable.
  The `kernel_entry!` macro generates the naked `_start` entry point.
- Host conveniences: `Error::io(op, source)` for wrapping `std::io` errors, and
  `TraceBuffer::string_entries()` for iterating decoded string log lines as
  `(hart_id, text)`.
- **Documentation**: an mdBook user and developer guide under `docs/`.

### Notes

- Minimum supported Rust version: **1.88** (naked functions, let-chains).
- The ET-SoC-1 is software-coherent: cross-hart sharing via fences alone does not
  propagate. The `spsc` demo probes this; a safe cache-operation and barrier layer
  for cross-hart sharing is planned for a future release.

## [0.1.0]

Initial release of the `et-rs` host crate (single crate; `et-abi` and `et-k-rs`
did not yet exist).
<!-- TODO: add the crates.io release date and the 0.1.0 feature set. -->

[0.3.0]: https://github.com/mathsDOTearth/et-rs/releases/tag/v0.3.0
[0.2.0]: https://github.com/mathsDOTearth/et-rs/compare/v0.2.0...v0.3.0
[0.1.0]: https://crates.io/crates/et-rs/0.1.0
