# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0/). The three crates
(`et-abi`, `et-rs`, `et-k-rs`) are released together and share a version.

## [0.2.0] - 2026-07-30

First public release to crates.io. (Versions 0.1.x were internal development only
and never published.)

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

[0.2.0]: https://github.com/mathsDOTearth/et-rs/releases/tag/v0.2.0
