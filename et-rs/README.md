# et-rs (`et_soc1`)

A host-side Rust crate for driving the Esperanto **ET-SoC-1** RISC-V accelerator.
It loads compute kernels, launches them on selected shires, moves results back
over DMA, and decodes on-device trace buffers. The device-ops protocol, the DRAM
allocator and the trace decoder are all pure Rust; the crate does not wrap the
vendor C++ runtime.

It runs against either real hardware (the PCIe kernel driver, default) or the SDK
software emulator (the `emu` feature), so a developer without a card can build
and test the same code that will drive the real device. See
[Hardware vs. the software emulator](#hardware-vs-the-software-emulator).

The public crate name is `et_soc1`; the package on disk is `et-rs`.

## Repository layout

This repository is a Cargo workspace of three crates that publish independently:

| Crate | Library | Side | Role |
|-------|---------|------|------|
| [`et-abi`](../et-abi/) | `et_abi` | shared (`no_std`) | Launch-argument structs defined once, used by both sides so they cannot drift on layout. |
| **`et-rs`** | `et_soc1` | host (`std`) | This crate: kernel load, launch, DMA, and the trace decoder. |
| [`et-k-rs`](../et-k-rs/) | `et_kernel` | device (`no_std`) | Library for writing compute kernels in Rust, plus demo kernels. |

`et-rs` and `et-abi` are host crates and form the workspace; `et-k-rs`
cross-compiles to `riscv64imac` with its own target configuration, so it is
excluded from the host workspace and built from its own directory. Both `et-rs`
and `et-k-rs` depend on `et-abi`; neither depends on the other.

## Requirements

This crate is not self-contained: it binds to the Esperanto ET-SoC-1 SDK and, in
its default configuration, drives a real device. Read this before building.

**Build time.** The FFI bindings are vendored (`src/bindings_*.rs`), so a plain
`cargo build` needs neither the SDK nor bindgen and works anywhere (including
docs.rs). The SDK is needed at build time only for the two optional paths:

* `--features regenerate-bindings` regenerates the vendored bindings from the SDK
  headers under `/opt/et` (or `ET_SDK_PREFIX`) and needs `libclang`; and
* `--features emu` compiles the C++ shim against the SDK (see below).

**Run time, default backend (real hardware).** `Device::open` talks to the PCIe
kernel driver, so you need:

* a physical ET-SoC-1 card with the `et` kernel driver loaded, exposing
  `/dev/etN_ops`, and
* permission to open and `ioctl` that device node.

There is no software fallback in the default build; on a machine without the card
`Device::open` fails.

**Run time, `emu` feature (software emulator, no card).** `Device::open_emulator`
drives the SDK's software emulator through a C++ shim, so you additionally need:

* a CMake toolchain and a C++ compiler (the shim is compiled during
  `cargo build --features emu`), and
* the SDK's C++ device-layer libraries and firmware ELFs (all under `/opt/et`).

This path needs no kernel driver and no hardware, which is what lets you develop
and test without a card. See
[Hardware vs. the software emulator](#hardware-vs-the-software-emulator).

## Using `et_soc1` as a dependency

Add to `Cargo.toml`:

```toml
[dependencies]
et-rs = "0.4"
```

To include the software-emulator backend (requires CMake and the SDK C++ libraries;
see [Requirements](#requirements)):

```toml
[dependencies]
et-rs = { version = "0.4", features = ["emu"] }
```

The package name is `et-rs`; the compiled library is named `et_soc1` (matching the
C/C++ SDK convention), so in Rust source files you write `use et_soc1::...`.

### Hardware backend example

The following is a minimal, self-contained programme that loads a RISC-V compute
kernel, launches it on a single shire with full user tracing, and prints every
decoded string entry from the trace buffer. It targets a real ET-SoC-1 card
(`/dev/et0_ops`).

```rust
use et_soc1::{Device, Error, LaunchOptions, TraceConfig};
use et_soc1::trace::{DecodedEntry, TraceBuffer};

fn main() -> et_soc1::Result<()> {
    let elf = std::fs::read("hello.elf").map_err(|e| Error::Io {
        op: "read kernel ELF",
        source: e,
    })?;

    // Open device 0 (/dev/et0_ops) and query its DRAM geometry.
    let device = Device::open(0)?;

    // Load the kernel ELF before calling alloc(), so the code lands at the DRAM
    // base, matching its link address.
    let kernel = device.load_kernel(&elf)?;

    // Allocate an 8 MiB trace buffer in device DRAM.
    let trace_buf = device.alloc(8 * 1024 * 1024)?;

    // Launch on shire 0 with a barrier and full user tracing.
    let opts = LaunchOptions::new(0x1)
        .with_trace(TraceConfig::full(trace_buf, 0x1))
        .with_args(vec![0u8; 64]);
    device.launch(&kernel, &opts)?;

    // Copy the trace buffer to host memory and decode it.
    let mut host = vec![0u8; trace_buf.size as usize];
    device.memcpy_d2h(trace_buf.addr, &mut host)?;

    if let Ok(tb) = TraceBuffer::parse(&host) {
        for entry in tb.entries() {
            if let DecodedEntry::String(s) = entry.decoded() {
                println!("[hart {}] {}", entry.hart_id, s.trim_end());
            }
        }
    }
    Ok(())
}
```

### Software-emulator backend example

With the `emu` feature, replace `Device::open` with `Device::open_emulator`.
No hardware or kernel driver is required; the emulator boots the SDK firmware
internally and presents the same API surface:

```rust
// Requires: et_soc1 = { version = "0.2", features = ["emu"] }
let device = et_soc1::Device::open_emulator("/opt/et", "/tmp/et-run")?;
// The remainder of the programme is identical to the hardware path above.
```

The `hello_sysemu` example in the repository exercises this path end to end.

## API at a glance

```rust
use et_soc1::{Device, LaunchOptions, TraceConfig};
use et_soc1::trace::{TraceBuffer, DecodedEntry};

let device = Device::open(0)?;                       // /dev/et0_ops, GET_USER_DRAM_INFO
let kernel = device.load_kernel(&elf_bytes)?;        // DMA-write PT_LOAD segments -> LoadedKernel
let trace  = device.alloc(8 * 1024 * 1024)?;         // DRAM bump allocator

let opts = LaunchOptions::new(0x1)                   // shire mask
    .with_trace(TraceConfig::full(trace, 0x1))       // full user tracing
    .with_args(vec![0u8; 64]);
device.launch(&kernel, &opts)?;                      // PUSH_SQ + POP_CQ

let mut host = vec![0u8; trace.size as usize];
device.memcpy_d2h(trace.addr, &mut host)?;           // DMA read-list command

for entry in TraceBuffer::parse(&host)?.entries() {  // pure-Rust et-trace decoder
    if let DecodedEntry::String(s) = entry.decoded() {
        println!("[hart {}] {}", entry.hart_id, s);
    }
}
```

`examples/hello.rs` is a complete, pure-Rust re-implementation of the SDK
`et-testdrive` "hello world":

```text
cargo run --example hello -- /path/to/hello.elf
```

### A device kernel in Rust too

[`et-k-rs/`](../et-k-rs/) is the **device-side** "hello world" written in pure
`no_std` Rust (no C), a drop-in replacement for the SDK's C `hello.c`. Build it
for the compute harts and run it with either example:

```bash
( cd et-k-rs && rustup target add riscv64imac-unknown-none-elf && cargo build --release )
K=et-k-rs/target/riscv64imac-unknown-none-elf/release/hello-rs

cargo run --features emu --example hello_sysemu -- "$K"   # emulator, no hardware
cargo run --example hello -- "$K"                         # real hardware
```

Both the host driver and the device kernel being Rust makes this an end-to-end
pure-Rust path from launch to decoded trace output.

## BLAS: single-precision GEMM

`et_soc1::blas::sgemm` launches the `sgemm-rs` device kernel to compute
C = alpha * A * B + beta * C using the ET-SoC-1 tensor extension. v0.1
supports alpha=1.0 and beta=0.0; N may be any positive integer (partial
last-column tiles are handled via the 64-byte-aligned row padding allocated
by `alloc_tensor_matrix`).

```rust,ignore
use et_soc1::{Device, Result, blas};

fn run() -> Result<()> {
    let elf    = std::fs::read("sgemm-rs").unwrap();
    let dev    = Device::open(0)?;
    let kernel = dev.load_kernel(&elf)?;

    let lda = (k * 4).next_multiple_of(64) as u32;  // 64-byte-aligned row stride
    let ldb = (n * 4).next_multiple_of(64) as u32;
    let ldc = (n * 4).next_multiple_of(64) as u32;

    let (a_addr, _) = blas::alloc_tensor_matrix(&dev, m, k)?;
    let (b_addr, _) = blas::alloc_tensor_matrix(&dev, k, n)?;
    let (c_addr, _) = blas::alloc_tensor_matrix(&dev, m, n)?;

    // ... upload A and B with dev.memcpy_h2d ...

    blas::sgemm(
        &dev, &kernel,
        m as u32, n as u32, k as u32,
        1.0, a_addr, lda,
             b_addr, ldb,
        0.0, c_addr, ldc,
        1,   // n_shires
    )?;
    Ok(())
}
```

`alloc_tensor_matrix` returns `(device_addr, lda_bytes)` with the row stride
padded to a 64-byte boundary, ready for the tensor load/store instructions.
See `examples/sgemm.rs` for the complete end-to-end demonstration (upload,
launch, download, and element-wise verification against a scalar reference).

```bash
( cd et-k-rs && cargo build --release )
K=et-k-rs/target/riscv64imac-unknown-none-elf/release/sgemm-rs
cargo run --manifest-path et-rs/Cargo.toml --release --example sgemm         -- "$K"  # 64x64x64
cargo run --manifest-path et-rs/Cargo.toml --release --example sgemm_partial -- "$K"  # 32x20x32 partial-N
```

## Concurrency demos

Two `et-k-rs` kernels explore Rust concurrency on the ET-SoC-1.

### Data-parallel reduction (`reduce-rs`) — fearless data parallelism

Sums a large array across a shire's **64 harts**, each reducing its **disjoint**
slice and writing its **own cache-line-padded** partial; the host combines them.

```bash
( cd et-k-rs && cargo build --release )
R=et-k-rs/target/riscv64imac-unknown-none-elf/release/reduce-rs
cargo run --features emu --example reduce -- "$R"   # emulator
cargo run            --example reduce -- "$R"       # real hardware -> RESULT PASS
```

The kernel body is **safe Rust** over the `Grid` abstraction: a hart can reach
only its own input slice and its own output cell, so an out-of-partition access
or a cross-hart data race is *unrepresentable* — data-race freedom guaranteed at
compile time, at zero runtime cost, with only a thin audited `unsafe` boundary.
It has no cross-hart sharing during the kernel, so it is **coherence-clean**:
verified on real hardware and under the emulator's memory-consistency checkers.
(Note: per-hart outputs are padded to a cache line — false sharing silently
corrupts data on this software-coherent chip.)

### Lock-free SPSC (`spsc-rs`) — a coherence probe

A single-producer/single-consumer, lock-free, **non-atomic** queue across two
harts (plain volatile loads/stores + `fence rw,rw`). It established that the
**ET-SoC-1 is software-coherent**: a fence-only queue does *not* propagate writes
between harts — even the two threads of one minion — so cross-hart shared memory
needs explicit cache management or a genuinely shared (uncached) region. The
algorithm itself is correct (it passes once memory is coherent). This is why the
reduction, which avoids cross-hart sharing entirely, is the solid headline demo.

## Architecture

| Layer | Module | Responsibility |
|-------|--------|----------------|
| High-level handle | `device` | DRAM bump allocator, kernel load, launch, DMA, trace extraction |
| BLAS launcher     | `blas`   | Host-side `sgemm` / `alloc_tensor_matrix`; validates arguments and calls `launch_spmd` |
| Command channel   | `transport` | `Transport` trait; `IoctlTransport` (`/dev/etN_ops`); `FfiTransport` (emulator, `emu` feature); `DmaHostBuffer` staging |
| Wire format       | `proto` | Device-ops command builders / response parsers, layout-asserted |
| Driver uapi       | `ioctl` | `ETSOC1_IOCTL_*` request codes and thin syscall wrappers |
| FFI               | `ffi` | bindgen output for `et_ioctl.h`, the device-ops enums and the et-trace layout |
| Trace             | `trace` | Standalone pure-Rust port of the reference `Trace_Decode` iterator |
| Emulator shim     | `emu-shim/` | C ABI over the SDK C++ device-layer's `DeviceSysEmu`, built by `build.rs` under the `emu` feature |

`Device` is generic over `Transport` (`Device<T: Transport = IoctlTransport>`),
so a backend is chosen without touching the API surface: `Device::open` for
hardware, `Device::open_emulator` for the emulator.

### Command mapping

| Method | Driver primitive |
|--------|------------------|
| `Device::open` | `open("/dev/etN_ops")` + `GET_USER_DRAM_INFO` |
| `load_kernel` | DMA-write each `PT_LOAD` segment to its link address (`DMA_WRITELIST` command); launch at `e_entry` |
| `alloc` | monotonic bump within the user DRAM region |
| `launch` | `device_ops_kernel_launch_cmd_t` -> `PUSH_SQ`, response via `POP_CQ` |
| `memcpy_h2d` / `memcpy_d2h` | `device_ops_dma_writelist_cmd_t` / `readlist_cmd_t` (DMA descriptor flag) -> `PUSH_SQ`/`POP_CQ` |
| `extract_cm_trace` | `GET_TRACE_BUFFER_SIZE` + `EXTRACT_TRACE_BUFFER` ioctl (`TRACE_BUFFER_CM`) |

`update_firmware` remains the `FW_UPDATE` path, kept separate because that ioctl
is for signed firmware images and rejects compute-kernel ELFs.

Two device-firmware contract details that the SDK headers state misleadingly, and
which `proto` therefore encodes explicitly (both verified against real firmware
via the emulator):

* `cmn_header.size` is the **total** command length, header included, not "the
  payload that follows the header" as the header comment says.
* DMA nodes need a valid host **physical** address; the emulator dereferences it
  directly, so DMA transfers stage through a `Transport::dma_host_buffer` (see
  [DMA staging](#dma-staging)).

## Building

The FFI bindings are vendored, so the default build needs no SDK:

```bash
cargo build                            # uses the committed src/bindings_*.rs
cargo test                             # off-device unit and integration tests
```

Normal builds never pass a feature flag: `cargo build` compiles the committed
`src/bindings_*.rs`, which are also shipped in the published crate, so most users
never regenerate at all.
 
Regenerate the vendored bindings **only when the SDK headers change**. This
**overwrites** the committed `src/bindings_*.rs` from the SDK it finds (so run it
deliberately, and against the intended SDK version), and needs the SDK and
`libclang`:
 
```bash
cargo build --features regenerate-bindings
ET_SDK_PREFIX=/opt/et cargo build --features regenerate-bindings   # explicit SDK
```

 Then review with `git diff` and commit the updated files. Once committed, plain
`cargo build` uses them again with no flag; the feature is not needed per machine,
only per header change.

See [Requirements](#requirements) for the runtime prerequisites of each backend.

### bindgen notes

The bindings are generated with bindgen (under `regenerate-bindings`) and then
committed; these notes explain the shape of the generated output.

* `et_ioctl.h` and `et-trace/layout.h` each define a conflicting
  `enum trace_buffer_type`, so they are generated in two separate translation
  units (`wrapper_ops.h`, `wrapper_trace.h`).
* The device-ops **message** structs are `packed, aligned(8)`, a combination
  bindgen 0.72 lowers to illegally nested `repr(packed)`/`repr(align)` types.
  Because `packed` is a layout no-op there (fields are already naturally
  aligned), those structs are transcribed by hand in `proto` with compile-time
  size assertions instead of being generated.
* bindgen cannot evaluate the function-like `_IOR`/`_IOW`/`_IOWR` macros, so the
  request codes are reconstructed in `ioctl` and pinned by a test against golden
  values emitted by the C preprocessor.

## Hardware vs. the software emulator

The crate runs against either real hardware or the SDK software emulator,
selected by the `Transport` backend:

* **Real hardware** (default): `Device::open(n)` uses `IoctlTransport` over
  `/dev/etN_ops`. Requires the `et` kernel driver loaded.
* **Software emulator** (`emu` feature): `Device::open_emulator(sdk_prefix,
  run_dir)` uses `FfiTransport`, which drives the vendor C++ device-layer's
  `DeviceSysEmu` backend through a small C ABI shim (`emu-shim/`). No hardware or
  kernel driver needed, so a developer without a card can build and test.

Because the crate builds the **same** device-ops command bytes for both, and only
the byte in/out differs, the `proto`, `device` and `trace` code paths are
exercised identically to the hardware path. The emulator run:

```bash
cargo run --features emu --example hello_sysemu -- ./path/to/hello.elf
# Booting software emulator (this can take a while)...
# Emulator ready: DRAM base 0x8005801000, size 17085497344 bytes; ...
# Kernel completed in 3738 cycles (waited 1500).
# [hart 0] Hello World from hart 0
# ... 64 harts ...
# Decoded 64 trace string entries.
```

Building the `emu` feature requires a CMake toolchain and the SDK C++ libraries;
the build script compiles the shim with the SDK's own `find_package` config so
the transitive link chain resolves automatically. The default build stays
pure-Rust (only `libc`) and needs none of this.

### DMA staging

DMA transfers do not use arbitrary host memory; each backend requires memory it
can actually reach, so `Device::memcpy_h2d`/`memcpy_d2h` (and thus
`load_kernel`) stage through a `Transport::dma_host_buffer`:

* `IoctlTransport` `mmap`s a buffer from the driver's CMA pool
  (`mmap(NULL, size, RW, MAP_SHARED, ops_fd, 0)`, one CMA allocation per
  mapping, as `DevicePcie::allocDmaBuffer` does). The driver resolves the bus
  address from the mapped virtual address, so the node's physical field is 0.
  A plain user buffer is rejected with `EINVAL`.
* `FfiTransport` uses the device-layer's `allocDmaBuffer`; the emulator
  dereferences the physical field directly, so both node fields carry the
  returned address.
* The default (`VecDmaBuffer`, used by test doubles) is a plain heap buffer.

### Off-device test coverage

Covered by `cargo test` without hardware or the emulator:

* the pure-Rust `trace` decoder, against synthetic buffers (including
  sub-buffer partitioning and malformed-input termination);
* the `proto` command builders and response parsers, via layout assertions and
  byte-level checks;
* the whole `device` layer (allocation, launch command construction, DMA
  splitting, trace extraction) through an in-memory `Transport` double;
* real-ELF kernel loading through the public API (when `hello.elf` is built);
* the `ioctl` request codes, pinned to values from the C header.

End-to-end validation runs on the emulator (`--features emu`, locally) and on
real hardware.

## Thanks

Thanks to AiNEKKO https://nekko.ai/ and AI Foundry https://aifoundry.org/ for allowing me 
time on their community ET-SoC-1 servers to develop this code.

The ET-SoC-1 ET Platform SDK and software emulator can be found on their GitHub: https://github.com/aifoundry-org/et-platform

## Licence

Apache-2.0, matching the ET Platform SDK headers this crate binds to.  
ET-SoC-1 ET Platform API is under the Apache 2 License.
