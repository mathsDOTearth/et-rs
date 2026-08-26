# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0/). The three crates
(`et-abi`, `et-rs`, `et-k-rs`) are released together and share a version.

## [0.5.0] - unreleased

### Added

- **`et-k-rs`**: `et_kernel::cache` -- L1 data cache management for
  software-coherent cross-hart sharing.
  - `CacheDest` enum: `L1`, `L2`, `L3`, `Mem`; selects how far up the cache
    hierarchy a writeback or eviction propagates.
  - `cache_writeback(addr, len)`: writes dirty L1 lines in the range
    `[addr, addr+len)` to DDR (CSR `0x8BF`, `flush_va`), then waits for
    completion via `TensorWait(CacheOp)` (PRM Section 8.1.3). Use on the
    producer side; call `fence()` before this function.
  - `cache_invalidate(addr, len)`: discards L1 lines in the range (CSR
    `0x89F`, `evict_va`), then waits via `TensorWait(CacheOp)`. Use on the
    consumer side after receiving the producer's synchronisation signal.
  - `cache_flush(addr, len)`: writeback then invalidate with a single
    `TensorWait(CacheOp)` at the end; safe when lines may be both dirty and
    stale.
  - `cache_writeback_to(dst, addr, len)` and `cache_invalidate_to(dst, addr,
    len)`: lower-level variants accepting an explicit `CacheDest`, e.g.
    `CacheDest::L2` for intra-shire coherence without a full DDR writeback.
  - `CSR_EVICT_VA = 0x89F`, `CSR_FLUSH_VA = 0x8BF`: raw CSR address
    constants.
  - Operations are processed in batches of up to 16 cache lines per hardware
    iteration (the 4-bit `NumLines-1` repeat field maximum); the range need not
    be 64-byte aligned.
- **`et-k-rs`**: `TensorEvent::CacheOp` (event code 6, PRM Table 9-2):
  "all previous cache management operations complete". Users who batch multiple
  cache ops and wish to issue a single `tensor_wait` at the end can use this
  variant with the lower-level `cache_writeback_to` / `cache_invalidate_to`
  primitives.
- **`et-k-rs`**: `cache-test-rs` -- hardware validation kernel that exercises
  `cache_writeback` across all 32 shires and 1024 Minions.
- **`et-rs`**: `examples/cache_test` -- host driver for the cache coherence
  test; verifies that every Minion's written value is visible to host DMA after
  `cache_writeback`.

### Fixed

- **`et-k-rs`**: `cache_writeback` / `cache_invalidate` / `cache_flush` and
  their `_to` variants now each issue `TensorWait(CacheOp)` (CSR `0x830`,
  event 6) after the `flush_va`/`evict_va` CSR writes. Per PRM Section 8.1.3,
  a `TensorWait` is required before any subsequent memory access to the
  affected lines; without it the cache op co-processor may still have
  outstanding traffic to DDR when the next instruction executes.
- **`et-k-rs`**: `cache-test-rs` kernel: restored `fence()` to after
  `cache_writeback` (matching the confirmed-passing sequence). Placing the
  fence before the call causes non-deterministic `EXCEPTION (status 2)` with
  random faulting-shire masks; Minion cores are in-order, so no pre-op fence
  is needed to drain the store buffer, and the post-op fence correctly orders
  the writeback completion relative to the `ecall` return.

### Changed

- **All crates**: `rust-version` raised from `1.88` to `1.98`. rustc 1.97.0
  contains a code-generation regression that produces incorrect instruction
  sequences for certain inline-assembly patterns (including the `flush_va` /
  `evict_va` CSR writes in this crate). Kernels compiled with 1.97.0 fault on
  hardware with `EXCEPTION (status 2)`. The bug is absent in 1.98.0.

## [0.4.2] - 2026-08-26

### Fixed

- **`et-k-rs`**: Added `tensor_wait(TensorEvent::Store)` before `fence()` in
  `compute_tile`. The tensor store DMA is asynchronous and runs independently
  of the RISC-V CPU; `fence rw,rw` alone does not drain it. Without the wait,
  the last tile's stores could be in-flight when the kernel returns via
  `ecall`, causing a race with the host's subsequent DMA read of C. Intermediate
  tiles were coincidentally safe (the next tile's `tensor_load` serialises at
  the co-processor), but the last tile had no implicit drain.

- **`et-rs`**: `sgemm` now validates `n_shires` is in `1..=63`. Previously
  `n_shires = 0` produced `shire_mask = 0` and launched on no shires, silently
  leaving C uninitialised; `n_shires >= 64` overflowed the `1_u64 << n_shires`
  shift used to build the shire mask.

### Changed

- **`et-abi`**: Corrected stale documentation on `GEMM_TILE_N`, `GemmArgs`
  layout invariants, and `GemmArgs::n` that still said "N must be a multiple
  of 16" -- this restriction was lifted in v0.4.0 and the docs were not updated.

- **`et-k-rs`**: Corrected stale comment in `compute_tile` that said "With N
  enforced as a multiple of GEMM_TILE_N" -- same v0.4.0 omission.

## [0.4.1] - 2026-08-26

### Fixed

- **`et-k-rs`** (RTLMIN-6496): `pmu_read`, `pmu_read_cycle`, `pmu_read_instret`,
  and `timestamp` now each emit four back-to-back `csrrs` reads of the same
  CSR in a `.align 4` (16-byte-aligned) block, discarding the first three
  results. This satisfies the RTLMIN-6496 erratum requirement and ensures
  the returned value is architecturally correct. Previously a single read was
  used, which returned an unreliable value on affected silicon.

### Added

- **`et-abi`**: `CachePadded<T>` -- a `#[repr(align(64))]` wrapper that places
  `T` on its own cache line. Use for per-hart output cells on the software-
  coherent ET-SoC-1 to prevent false-sharing corruption without explicit cache
  operations. Derives `Clone`, `Copy`, `Debug`, `Default`, `PartialEq`, `Eq`;
  the inner value is accessed via the public field `.0`.

- **`et-rs`**: `DeviceProperties` struct -- exposes all thirteen fields of the
  driver's `dev_config` descriptor, including `minion_boot_freq` (MHz) for
  cycle-to-microsecond conversion, cache sizes, DDR bandwidth, form factor,
  TDP, L2 bank count, sync-minion shire ID, architecture revision, and device
  number. Obtain via `Device::properties()`.

- **`et-rs`**: `Device::properties() -> Result<DeviceProperties>` method.

- **`et-rs`**: `Transport::device_properties()` trait method with a default
  implementation that fills conservative values; `IoctlTransport` overrides it
  to issue a single `GET_DEVICE_CONFIGURATION` ioctl and return all fields.

### Changed

- **`et-rs`**: `Device::alloc` now aligns to `max(dma_alignment, CACHE_LINE)`
  (64 bytes minimum) rather than `dma_alignment` alone. This prevents two
  caller-allocated regions from sharing a cache line, eliminating a potential
  source of false-sharing corruption on this software-coherent architecture
  regardless of the device's reported alignment.

## [0.4.0] - 2026-08-24

### Added

- **`et-k-rs`**: `et_kernel::pmu` -- Performance Monitoring Unit counter API.
  - `PmuEvent` enum: typed event codes for `mhpmeventN` assignment, including
    `TfmaWaitTenb = 18` (PRM Chapter 8; measures cycles spent waiting for
    TenB load before TensorFMA32).
  - `pmu_read(counter: u8) -> u64`: reads `hpmcounterN` (CSR `0xC03 + (N-3)`)
    in U-mode; safe to call from any kernel. Covers counters 3..=31.
  - `pmu_read_cycle() -> u64`: reads the `cycle` CSR (`0xC00`).
  - `pmu_read_instret() -> u64`: reads the `instret` CSR (`0xC02`).

- **`et-k-rs`**: `tensor_load_l2` -- L2 prefetch intrinsic (CSR `0x85F`,
  TensorLoadL2Scp). Loads rows from memory into the shire L2 cache without
  consuming any L1 scratchpad lines. Issue this for the next A row-tile while
  the current k-loop FMA executes; the subsequent `tensor_load` (L1 fill)
  completes from L2 rather than DRAM, removing A-DMA latency from the critical
  path. `CSR_TENSOR_LOAD_L2 = 0x85F` is exported alongside the other CSR
  constants.

- **`et-k-rs`**: `TensorError` -- typed tensor co-processor error status.
  `check_tensor_error() -> Result<(), TensorError>` calls `tensor_error()` and
  returns `Ok(())` when no fault is latched or `Err(TensorError)` otherwise.
  `TensorError::raw()` exposes the raw CSR value; named bit accessors for the
  PRM Table 9-3 error flags will be added once bit positions are confirmed on
  hardware. `tensor_error()` is now `#[must_use]`.

### Changed

- **`et-k-rs`**: `sgemm-rs` kernel switches from global-cyclic to
  **shire-blocked** tile distribution. Each shire now handles a contiguous
  `ceil(n_tiles / n_shires)` slice of the tile grid; within the block, the 32
  Minions distribute cyclically with step 32. Concentrating all Minions in a
  shire on the same row-band of C improves A-row reuse in the shire-shared L2
  cache. Hardware-measured improvement: +26% throughput at N=4096.
  No API change; output is bit-for-bit identical.

- **`et-rs`**: `sgemm` now accepts **arbitrary N** (any positive integer). The
  kernel computes `ceil(N / GEMM_TILE_N)` column tiles; the last partial tile
  uses the row-stride padding already allocated by `alloc_tensor_matrix`, so
  no additional padding is required from the caller. Only C[row][0..N] is
  meaningful on output; padding bytes are overwritten with partial-tile FMA
  results. `GemmError::NNotMultipleOfTileN` is retained for source
  compatibility but marked `#[deprecated(since = "0.4.0")]` and is never
  returned by `sgemm`.

## [0.3.1] - 2026-08-23

### Fixed

- **`et-k-rs`**: `tensor_load_b` was missing the `id: bool` parameter
  (PRM Chapter 9: bit 0 of x31 selects the load event ID for both
  TensorLoad and TensorLoadB). Without it, every B load generated a
  `Load0` event, making `tensor_wait(Load0)` serialise on both A and B
  DMAs instead of A alone. The parameter is added as the final argument
  (`id: bool`); pass `true` to use `Load1`, keeping B's event independent
  of A's. The `sgemm-rs` call site is updated accordingly.
- **`et-k-rs`**: `TensorEvent::Store = 8` was absent from the enum
  (PRM Table 9-2). Without it, callers could not issue a targeted
  `tensor_wait(Store)` to drain only the tensor store DMA; the only
  alternative was the broader `fence rw, rw`. The variant is now present;
  no change to `tensor_wait` is needed.

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

[0.5.0]: https://github.com/mathsDOTearth/et-rs/compare/v0.4.2...v0.5.0
[0.4.0]: https://github.com/mathsDOTearth/et-rs/releases/tag/v0.4.0
[0.3.1]: https://github.com/mathsDOTearth/et-rs/compare/v0.3.1...v0.4.0
[0.3.0]: https://github.com/mathsDOTearth/et-rs/compare/v0.3.0...v0.3.1
[0.2.0]: https://github.com/mathsDOTearth/et-rs/compare/v0.2.0...v0.3.0
[0.1.0]: https://crates.io/crates/et-rs/0.1.0
