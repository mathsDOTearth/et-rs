# et-k-rs

**Device-side library for writing ET-SoC-1 compute kernels in pure `no_std`
Rust** — the device counterpart to the `et_soc1` host crate, with no C
dependency. The library (`et_kernel`, `src/lib.rs`) provides hart identity, the
U-mode trace write, a hardware `fence`, scratchpad addressing, and the safe
`Grid` partitioning abstraction. Launch-argument structs are shared with the host
launcher through the [`et-abi`](../et-abi) crate, so the two sides cannot drift
on layout. Three demo kernels build on the library and double as worked examples.

## Kernels

- **`hello-rs`** (`src/bin/hello.rs`) — every hart writes `"Hello World from
  hart N"` to its trace buffer. A drop-in Rust replacement for the SDK's C
  `hello.c`; reimplements `get_hart_id` (the `hartid` CSR `0xCD0`) and the
  `Trace_String` write directly.
- **`spsc-rs`** (`src/bin/spsc.rs`) — a single-producer/single-consumer,
  lock-free, **non-atomic** queue across two harts (plain volatile loads/stores +
  `fence rw,rw`, no atomics, no locks). A coherence *probe*: it showed the
  ET-SoC-1 is **software-coherent** — fence-only cross-hart sharing does not
  propagate (even within one minion), so this needs explicit cache management or
  genuinely shared memory. See the crate root README.
- **`reduce-rs`** (`src/bin/reduce.rs`) — a **data-parallel reduction** (sum)
  over a DRAM array across a shire's 64 harts. Each hart reduces its **disjoint**
  slice (`Grid::my_slice`) and writes its **own cache-line-padded** partial cell
  (no false sharing); the host combines. No cross-hart sharing during the kernel,
  so it is coherence-clean and validated on hardware.

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
K=et-k-rs/target/riscv64imac-unknown-none-elf/release
cargo run --features emu --example hello_sysemu -- $K/hello-rs   # emulator
cargo run            --example reduce        -- $K/reduce-rs     # hardware
cargo run            --example spsc          -- $K/spsc-rs       # hardware
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
