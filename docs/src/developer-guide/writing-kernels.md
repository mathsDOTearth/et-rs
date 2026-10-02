# Writing kernels

A compute kernel is a freestanding `no_std` RISC-V binary that links against the
`et_kernel` library. The demo kernels in `et-k-rs/src/bin/` (`hello-rs`,
`spsc-rs`, `reduce-rs`, `sgemm-rs`, `cache-test-rs`, `tensor-ext-test`,
`simd-test-rs`) are worked examples; this page covers what every kernel needs.

## Build configuration

`et-k-rs/.cargo/config.toml` fixes the target and code model for the whole crate:

```toml
[build]
target = "riscv64imac-unknown-none-elf"

[target.riscv64imac-unknown-none-elf]
rustflags = [
    "-C", "code-model=medium",       # rustc's name for RISC-V medany (PC-relative)
    "-C", "relocation-model=static",
    "-C", "target-feature=+f",       # single-precision F extension
]
```

`build.rs` passes the linker script (`-T link.ld`) as an absolute path.

The ET-Minion implements RV64IMAC plus F, with the FP registers widened to 256
bits for the PS SIMD extension; it has no D extension. `riscv64gc` must not be
used: it implies D and the LP64D ABI, under which the compiler saves the
callee-saved registers `fs0`-`fs11` with `fsd`/`fld` around any function whose
inline assembly clobbers them, and those instructions trap as illegal on every
hart. Under `riscv64imac` with `+f` the ABI remains LP64, no FP register is
callee-saved, and no such saves are emitted. rustc warns that `f` is an
unstable target feature; the warning is cosmetic and cannot be suppressed.

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
unsafe { tensor_store(c_addr, arows, ldc); tensor_wait(TensorEvent::Store); }
fence();
```

The `tensor_wait(TensorEvent::Store)` is required: the tensor store DMA runs
independently of the hart, and `fence` does not wait for it.

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

## PS SIMD extension

`et_kernel::simd` provides wrappers for the ET-SoC-1 packed-single (PS) SIMD
extension, hardware-verified on aifoundry3 (2026-09-18). PS instructions operate
on the standard RISC-V FP register file (f0..f31), treating each 256-bit register
as a vector of eight f32 lanes.

| Function | PS instruction | Role |
|---|---|---|
| `broadcast_ps_bits(bits, dest)` | `FBCX.PS` | Broadcasts a raw 32-bit pattern to all 8 lanes of `f[dest]`. |
| `broadcast_ps(scalar, dest)` | `FBCX.PS` | Broadcasts an `f32` to all 8 lanes of `f[dest]` (via `to_bits`). |
| `fmul_ps_row(row, scratch)` | `FMUL.PS` x 2 | Multiplies `f[2*row]` and `f[2*row+1]` element-wise by pre-broadcast `f[scratch]`. |
| `scale_c_row(row, alpha, scratch)` | `FBCX.PS` + `FMUL.PS` x 2 | Convenience wrapper: broadcast `alpha` into `f[scratch]`, then scale the row. |
| `PS_SCRATCH_DEFAULT` | -- | `28` (f28/ft8): safe scratch register for tiles of at most 14 rows. |
| `load_ps(register, addr)` | `FLQ2` | Loads 8 consecutive f32 (32 B, 32-byte aligned) into `f[register]`; unmasked. |
| `store_ps(register, addr)` | `FSQ2` | Stores all 8 lanes of `f[register]`; unmasked. Write back with `cache_writeback` for the host. |
| `fexp_ps(register)` | `FEXP.PS` | In place, `2^x` per lane. |
| `flog_ps(register)` | `FLOG.PS` | In place, `log2 x` per lane. |
| `frcp_ps(register)` | `FRCP.PS` | In place, `1/x` per lane. |

`FEXP.PS`, `FLOG.PS` and `FRCP.PS` execute natively, within 1 ULP with
round-towards-zero; subnormal inputs are treated as zero and subnormal results
flushed to zero (PRM). For the natural exponential, scale the argument by
log2(e) first. `FDIV.PS`, `FSQRT.PS`, `FRSQ.PS` and `FSIN.PS` are deliberately
not wrapped: they trap to M-mode emulation and cost far more than a native
instruction. The `ps-math-test-rs` kernel with the `ps_math_test` example
measures the error of every lane on hardware.

The typical pattern for scaling a row of a C tile by alpha:

```rust,ignore
use et_kernel::simd::{PS_SCRATCH_DEFAULT, scale_c_row};

// Scale rows 0..n_rows of the FP register C tile by alpha.
for row in 0..n_rows {
    unsafe { scale_c_row(row, alpha, PS_SCRATCH_DEFAULT); }
}
```

For a full 16-row tile, rows 14 and 15 occupy f28/f29 and f30/f31, which
conflict with `PS_SCRATCH_DEFAULT` (f28). A scalar spill does not help: `fsw`
and `fsd` save only the low 32 or 64 bits (lane 0, or lanes 0-1) of a 256-bit
register. Spill f28 with the full-width `store_ps` to a 32-byte-aligned slot,
broadcast and scale, then restore it with `load_ps`.

Each wrapper declares all 32 FP registers clobbered, so the compiler keeps
none of its own values in f0..f31 across a call. The C-tile contents are,
however, invisible to the compiler: code between the FMA that produced the tile
and the final `tensor_store` must not itself use floating point, or the
compiler may allocate a register that holds tile data.

### Feature gate

Stable rustc does not expose the RISC-V `f`/`d` target features to `cfg`, even
when they are enabled, so the module cannot be gated on `target_feature = "f"`.
Instead `et-k-rs/build.rs` emits `cfg(et_fp_registers)` when the target triple's
ISA string includes F (`g`, `f` or `d`) or `RUSTFLAGS` enables `+f`/`+d`. The
`simd` module and the FP-clobbering form of the internal `fp_asm!` macro are
compiled only under that cfg; on a target without FP registers the module is
empty and call sites must be gated likewise.

## Device facts (reference)

- `hart_id` reads the custom `hartid` CSR `0xCD0` (not `mhartid` `0xF14`).
- A cycle timestamp reads `hpmcounter3`, CSR `0xC03`; `pmu_read(3)` is
  equivalent. `pmu_read_cycle()` reads the `cycle` CSR (`0xC00`).
- `trace_str` writes a string entry into the per-hart trace control block; the
  firmware finalises the sub-buffer size headers on kernel return, after which the
  host decodes them with `et_soc1::trace`.
- No heap, no `.bss`, no unwinding.
