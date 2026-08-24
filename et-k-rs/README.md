# et-k-rs

**Device-side library for writing ET-SoC-1 compute kernels in pure `no_std`
Rust** — the device counterpart to the `et_soc1` host crate, with no C
dependency. The library (`et_kernel`, `src/lib.rs`) provides hart identity, the
U-mode trace write, a hardware `fence`, scratchpad addressing, and the safe
`Grid` partitioning abstraction. Launch-argument structs are shared with the host
launcher through the [`et-abi`](../et-abi) crate, so the two sides cannot drift
on layout. Three demo kernels build on the library and double as worked examples.

## Kernels

- **`hello-rs`** (`src/bin/hello.rs`) -- every hart writes `"Hello World from
  hart N"` to its trace buffer. A drop-in Rust replacement for the SDK's C
  `hello.c`; reimplements `get_hart_id` (the `hartid` CSR `0xCD0`) and the
  `Trace_String` write directly.
- **`spsc-rs`** (`src/bin/spsc.rs`) -- a single-producer/single-consumer,
  lock-free, **non-atomic** queue across two harts (plain volatile loads/stores +
  `fence rw,rw`, no atomics, no locks). A coherence *probe*: it showed the
  ET-SoC-1 is **software-coherent** -- fence-only cross-hart sharing does not
  propagate (even within one minion), so this needs explicit cache management or
  genuinely shared memory. See the crate root README.
- **`reduce-rs`** (`src/bin/reduce.rs`) -- a **data-parallel reduction** (sum)
  over a DRAM array across a shire's 64 harts. Each hart reduces its **disjoint**
  slice (`Grid::my_slice`) and writes its **own cache-line-padded** partial cell
  (no false sharing); the host combines. No cross-hart sharing during the kernel,
  so it is coherence-clean and validated on hardware.
- **`sgemm-rs`** (`src/bin/sgemm.rs`) -- single-precision GEMM (C = A*B) using
  the ET-SoC-1 tensor extension. Tile assignment is **shire-blocked**: each shire
  handles a contiguous slice of the tile grid, improving A-row reuse in the
  shire-shared L2 (+26% at N=4096 over global-cyclic). v0.1 supports alpha=1.0
  and beta=0.0; N may be any positive integer (partial last-column tile handled
  via stride-aligned padding). Verified on hardware: 64x64x64 and 32x20x32
  (partial-N) both produce correct results with zero floating-point error. See
  the host-side `sgemm` and `sgemm_partial` examples in `et-rs/`.

## Tensor extension (`et_kernel::tensor`)

All tensor operations on the ET-SoC-1 are encoded as standard RISC-V
`csrrw xd, <csr>, xs` writes (PRM Chapter 9). No custom opcode or target
feature is required; `riscv64imac` suffices. The `tensor` module exposes
typed, inline-asm wrappers for each instruction:

| Function | CSR | Role |
|---|---|---|
| `tensor_load` | `0x83F` | Async load from DRAM into L1 scratchpad (ID=0 or ID=1). |
| `tensor_load_b` | `0x83F` | Async load from DRAM into the TenB register file (bit 52 set). |
| `tensor_load_l2` | `0x85F` | Async prefetch from DRAM to shire L2 cache (no L1 fill). |
| `tensor_fma32` | `0x801` | Async FMA32: C += A * B (or C = A * B when `mul_only`). |
| `tensor_store` | `0x87F` | Async store from FP register file to DRAM. |
| `tensor_wait` | `0x830` | Stall hart until `Load0`, `Load1`, `Fma`, or `Store` event fires. |
| `tensor_error` | `0x808` | Read latched co-processor error flags (`#[must_use]`). |
| `check_tensor_error` | `0x808` | Returns `Ok(())` or `Err(TensorError)` (typed wrapper). |
| `set_tensor_mask` | `0x805` | Write per-row FMA enable bits. |

`fma32_xs` constructs the `xs` bit field for `tensor_fma32`, encoding BCOLS,
AROWS, ACOLS, AOFFSET, TENB, BSTART, ASTART, MUL, and MSK from PRM Table 9-4.

x31 (`t6`) carries the row stride for `tensor_load`, `tensor_load_b`, and
`tensor_store`; each function sets it atomically with the `CSRRW` inside the
same asm block.

### Usage pattern

```rust,ignore
use et_kernel::tensor::{
    TensorEvent, check_tensor_error, fma32_xs,
    tensor_fma32, tensor_load, tensor_load_b, tensor_store, tensor_wait,
};
use et_kernel::fence;

// Load A tile into L1 scratchpad (lines 0..arows); ID=false -> Load0.
unsafe { tensor_load(a_addr, 0, arows, /*id=*/false, lda); }
unsafe { tensor_wait(TensorEvent::Load0); }

// Load B tile into TenB register file; ID=true -> Load1, keeping B's
// event independent of the A Load0 above.
unsafe { tensor_load_b(b_addr, acols, /*coop=*/false, ldb, /*id=*/true); }

// Issue FMA: C = A * B (first k-tile) or C += A * B (subsequent).
let xs = fma32_xs(bcols, arows, acols, 0, true, 0, 0, k_tile == 0, false);
unsafe { tensor_fma32(xs); tensor_wait(TensorEvent::Fma); }

// Optional: check for co-processor faults before storing.
check_tensor_error().expect("tensor co-processor fault");

// Store C from FP registers to DRAM, then drain the store and fence.
unsafe { tensor_store(c_addr, arows, ldc); }
unsafe { tensor_wait(TensorEvent::Store); }
fence();
```

## PMU counters (`et_kernel::pmu`)

U-mode-accessible hardware performance counters for characterising kernel
behaviour. Reads `hpmcounterN` (CSR `0xC03 + (N-3)`) without any privilege
escalation.

```rust,ignore
use et_kernel::pmu::{PmuEvent, pmu_read};

// Read counter 4 before and after the k-loop; the delta is the number of
// TFMA_WAIT_TENB stall cycles (assuming firmware assigned PmuEvent::TfmaWaitTenb
// to counter 4 via mhpmevent4).
let before = pmu_read(4);
// ... tensor k-loop ...
let after  = pmu_read(4);
let stalls = after.wrapping_sub(before);
```

| Function | Description |
|---|---|
| `pmu_read(counter: u8) -> u64` | Read `hpmcounterN` for N in 3..=31. |
| `pmu_read_cycle() -> u64` | Read `cycle` CSR (`0xC00`). |
| `pmu_read_instret() -> u64` | Read `instret` CSR (`0xC02`). |
| `PmuEvent::TfmaWaitTenb = 18` | Cycles stalled waiting for TenB load (PRM Ch. 8). |

## PS SIMD stub (`et_kernel::simd`)

The `simd` module provides placeholder wrappers for the ET-SoC-1 packed-single
(PS) SIMD extension, gated on `cfg(target_feature = "f")`. The stubs
(`scale_c_row`, `broadcast_ps`) compile and link but contain no real asm; the
opcode encodings must be confirmed from PRM Chapter 5 before the bodies are
filled in. The module is marked `#[doc(hidden)]` and excluded from published
documentation. Do not depend on it in production code.

## The safety story (`reduce-rs`)

The kernel body is safe Rust over `Grid`: a hart can obtain only its own input
slice and its own output cell, so an out-of-partition access or a cross-hart data
race is *unrepresentable*. The only `unsafe` is a thin, commented boundary that
turns launch arguments and device addresses into typed slices.

## Build

Cross-compiles to the compute harts (RV64IMAC); target, code model
(`medium` = medany, for the fixed high link address) and linker script are in
`.cargo/config.toml`:

```bash
rustup target add riscv64imac-unknown-none-elf   # once
cargo build --release
# -> target/riscv64imac-unknown-none-elf/release/{hello-rs,spsc-rs,reduce-rs}
```

## Run

Load and launch with the host crate's examples (from the repository root),
emulator or hardware:

```bash
cd et-k-rs && cargo build --release && cd ..
K=et-k-rs/target/riscv64imac-unknown-none-elf/release
cargo run --manifest-path et-rs/Cargo.toml --release --example hello_sysemu --features emu -- $K/hello-rs   # emulator
cargo run --manifest-path et-rs/Cargo.toml --release --example reduce         -- $K/reduce-rs  # hardware
cargo run --manifest-path et-rs/Cargo.toml --release --example spsc           -- $K/spsc-rs    # hardware
cargo run --manifest-path et-rs/Cargo.toml --release --example sgemm          -- $K/sgemm-rs   # hardware; 64x64x64
cargo run --manifest-path et-rs/Cargo.toml --release --example sgemm_partial  -- $K/sgemm-rs   # hardware; 32x20x32 partial-N
```

## Kernel facts (reference)

- **Entry/exit:** `_start` sets `gp`, calls `entry_point`, then `ecall` with
  `SYSCALL_RETURN_FROM_KERNEL` (8) / `KERNEL_RETURN_SUCCESS` (0). Firmware sets
  the stack pointer.
- **Launch args:** the launch command's `pointer_to_args` is delivered in `a0`
  (not `ra`, despite the SDK docs — verified on device); `a0` flows through
  `_start` into `entry_point`'s first parameter.
- No `.bss` (the linker script asserts it), no heap, no unwinding
  (`panic = "abort"`).

## Publishing

This crate is a separate cargo package from the host crate (and excluded from the
host workspace) because it targets RISC-V bare metal. Its library and demo bins
use RISC-V inline assembly and a linker script, so they cannot be built for the
host target; a crates.io release must therefore verify against the device target:

```bash
cargo publish -p et-abi                                        # dependency first
cargo publish --target riscv64imac-unknown-none-elf            # from et-k-rs/
```

## Thanks

Thanks to AiNEKKO https://nekko.ai/ and AI Foundry https://aifoundry.org/ for allowing me 
time on their community ET-SoC-1 servers to develop this code.

The ET-SoC-1 ET Platform SDK and software emulator can be found on their GitHub: https://github.com/aifoundry-org/et-platform

## Licence

Apache-2.0, matching the ET Platform SDK headers this crate binds to.  
ET-SoC-1 ET Platform API is under the Apache 2 License.