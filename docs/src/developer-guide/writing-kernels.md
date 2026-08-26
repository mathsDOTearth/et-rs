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

Each kernel provides a tiny `_start` (a naked function) that sets the global
pointer, calls the Rust entry point, and returns to the firmware via `ecall`. It
is placed in `.text.init`, which the linker script lays down first at the entry
address:

```rust,ignore
use core::arch::naked_asm;

#[unsafe(naked)]
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.init")]
pub extern "C" fn _start() -> ! {
    naked_asm!(
        ".option push",
        ".option norelax",
        "la gp, __global_pointer$",
        ".option pop",
        // a0 (the firmware-provided args pointer) passes straight through.
        "call entry_point",
        "li a2, 0",  // KERNEL_RETURN_SUCCESS
        "mv a1, a0", // return value
        "li a0, 8",  // SYSCALL_RETURN_FROM_KERNEL
        "ecall",
    )
}
```

The launch command's `pointer_to_args` arrives in **`a0`**, which flows straight
through `call entry_point` into the Rust function's first argument. (The SDK docs
say `ra`; on the device `ra` is 0 at entry.) Firmware sets the stack pointer.
Naked functions require Rust 1.88 (the crate MSRV).

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

## Tensor extension

The ET-SoC-1 tensor co-processor is accessible from any kernel through the
`et_kernel::tensor` module. All tensor instructions are standard RISC-V
`csrrw` writes; no custom target feature is needed. The typical pattern is:

```rust,ignore
use et_kernel::tensor::{TensorEvent, fma32_xs, tensor_fma32,
                        tensor_load, tensor_load_b, tensor_store, tensor_wait};
use et_kernel::fence;

// Only the primary hart (mhartid & 1 == 0) issues tensor instructions.
unsafe { tensor_load(a_addr, 0, arows, false, lda); }
unsafe { tensor_wait(TensorEvent::Load0); }
unsafe { tensor_load_b(b_addr, acols, false, ldb); }
let xs = fma32_xs(bcols, arows, acols, 0, true, 0, 0, /*mul_only=*/true, false);
unsafe { tensor_fma32(xs); tensor_wait(TensorEvent::Fma); }
unsafe { tensor_store(c_addr, arows, ldc); }
fence();
```

For CSR addresses, the x31 stride convention, the `fma32_xs` bit field, and a
full worked example, see the [Tensor extension](tensor-extension.md) page.

## PMU counters

`et_kernel::pmu` exposes hardware performance counters readable from U-mode.
Use `pmu_read(counter)` to read `hpmcounterN` (CSR `0xC03 + (N-3)`, N in 3..=31),
and `PmuEvent::TfmaWaitTenb = 18` to identify the TenB-wait stall event (PRM
Chapter 8). Useful for measuring B-load serialisation cost in the k-loop.

**RTLMIN-6496.** All PMU read functions -- `pmu_read`, `pmu_read_cycle`,
`pmu_read_instret`, and `timestamp` -- apply the RTLMIN-6496 hardware erratum
workaround automatically: four consecutive reads of the same CSR in a
16-byte-aligned block, returning only the fourth value. No call-site changes
are required; the workaround is invisible to the caller.

```rust,ignore
use et_kernel::pmu::{pmu_read, PmuEvent};
let before = pmu_read(4);
// ... tensor operations ...
let stalls = pmu_read(4).wrapping_sub(before);
```

## Device facts (reference)

- `hart_id` reads the custom `hartid` CSR `0xCD0` (not `mhartid` `0xF14`).
- A cycle timestamp reads `hpmcounter3`, CSR `0xC03`; `pmu_read(3)` is
  equivalent. `pmu_read_cycle()` reads the `cycle` CSR (`0xC00`).
- `trace_str` writes a string entry into the per-hart trace control block; the
  firmware finalises the sub-buffer size headers on kernel return, after which the
  host decodes them with `et_soc1::trace`.
- No heap, no `.bss`, no unwinding.
