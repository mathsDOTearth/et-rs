# Writing kernels

A compute kernel is a freestanding `no_std` RISC-V binary that links against the
`et_kernel` library. The three demo kernels in `et-k-rs/src/bin/` are worked
examples; this page covers what every kernel needs.

## Build configuration

`et-k-rs/.cargo/config.toml` fixes the target and code model for the whole crate:

```toml
[build]
target = "riscv64imac-unknown-none-elf"

[target.riscv64imac-unknown-none-elf]
rustflags = [
    "-C", "code-model=medium",       # rustc's name for RISC-V medany (PC-relative)
    "-C", "link-arg=-Tlink.ld",      # our linker script places the image
    "-C", "relocation-model=static",
]
```

The kernel is linked at a fixed high U-mode address (`0x8005801000`) that
coincides with the base of the user DRAM region, so the `medany` code model is
required. `link.ld` sets the entry to `_start`, places the image at that address,
and asserts there is no `.bss` (the kernel has no zero-init data segment). The
release profile uses `panic = "abort"`: there is no unwinding.

## Entry and exit

Each kernel provides a tiny `_start` that sets the global pointer, calls the Rust
entry point, and returns to the firmware via `ecall`:

```rust,ignore
global_asm!(
    ".section .text.init, \"ax\"",
    ".global _start",
    "_start:",
    "    la gp, __global_pointer$",
    // a0 (the firmware-provided args pointer) passes straight through.
    "    call entry_point",
    "    li a2, 0",          // KERNEL_RETURN_SUCCESS
    "    mv a1, a0",         // return value
    "    li a0, 8",          // SYSCALL_RETURN_FROM_KERNEL
    "    ecall",
);
```

The launch command's `pointer_to_args` arrives in **`a0`**, which flows straight
through `call entry_point` into the Rust function's first argument. (The SDK docs
say `ra`; on the device `ra` is 0 at entry.) Firmware sets the stack pointer.

## Arguments: the shared ABI

Read arguments through the struct defined once in `et-abi`, so host and device
cannot disagree on layout:

```rust,ignore
use et_abi::{DeviceArgs, ReduceArgs};

#[unsafe(no_mangle)]
pub extern "C" fn entry_point(args_ptr: usize) -> i64 {
    // SAFETY: firmware passed the launch command's pointer_to_args in a0.
    let args = unsafe { ReduceArgs::from_ptr(args_ptr as *const u8) };
    // ...
}
```

## The `Grid` abstraction and data safety

Launch is SPMD: every hart of every selected shire runs the kernel. `et_kernel`'s
`Grid` turns this into safe data parallelism. Constructed with the participating
hart count, it gives each hart only:

- `my_slice(data)`: its disjoint sub-slice of a shared input, and
- `output_cell(base)`: its own cache-line-padded output cell.

Because a hart cannot name another hart's slice or cell, an out-of-partition
access or a cross-hart data race is unrepresentable in safe kernel code. The only
`unsafe` is the boundary that turns raw device addresses into typed slices
(`device_slice`). The cache-line padding of output cells is not optional: see the
[Coherence model](coherence-model.md).

## Device facts (reference)

- `hart_id` reads the custom `hartid` CSR `0xCD0` (not `mhartid` `0xF14`).
- A cycle timestamp reads `hpmcounter3`, CSR `0xC03`.
- `trace_str` writes a string entry into the per-hart trace control block; the
  firmware finalises the sub-buffer size headers on kernel return, after which the
  host decodes them with `et_soc1::trace`.
- No heap, no `.bss`, no unwinding.
