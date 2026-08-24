# Tensor extension

The ET-SoC-1 tensor co-processor accelerates dense matrix operations. Every
tensor instruction is a standard RISC-V `csrrw xd, <csr>, xs` write (PRM
Chapter 9, Tables 9-1 and 9-7); no custom opcode or non-standard target
feature is required. The `riscv64imac` target and the stable Rust toolchain
suffice.

## CSR address map

The verified addresses (some SDK examples used incorrect values that cause an
ILLEGAL INSTRUCTION exception):

| Constant | CSR name | Address | Notes |
|---|---|---|---|
| `CSR_TENSOR_FMA` | `tensor_fma` | `0x801` | bits 3:1 select variant; 000 = FMA32 |
| `CSR_TENSOR_WAIT` | `tensor_wait` | `0x830` | stalls hart until the event fires |
| `CSR_TENSOR_ERROR` | `tensor_error` | `0x808` | latched co-processor error flags (`#[must_use]`) |
| `CSR_TENSOR_MASK` | `tensor_mask` | `0x805` | per-row FMA enable bits |
| `CSR_TENSOR_STORE` | `tensor_store` | `0x87F` | store from FP registers to DRAM |
| `CSR_TENSOR_LOAD` | `tensor_load` | `0x83F` | load to scratchpad (bit 52=0) or TenB (bit 52=1) |
| `CSR_TENSOR_LOAD_L2` | `tensor_load_l2` | `0x85F` | prefetch to shire L2 without L1 scratchpad fill |

All constants are defined in `et_kernel::tensor` and marked `pub`.

## Concurrency model

The tensor co-processor runs asynchronously alongside the RISC-V integer
pipeline. The hart must call `tensor_wait` before reading results or reusing
resources:

- `tensor_wait(TensorEvent::Load0)` before `tensor_fma32`: guarantees that
  scratchpad lines filled by the preceding `tensor_load` are visible to the FMA.
- `tensor_wait(TensorEvent::Fma)` before `tensor_store`: guarantees that the FP
  register file holds the final accumulated values.
- `tensor_wait(TensorEvent::Store)` drains only the tensor store DMA, allowing
  non-tensor scalar work to be interleaved (e.g. pointer arithmetic for the next
  tile). `fence()` is still required when other agents or the DMA engine must
  observe the stores.
- `check_tensor_error() -> Result<(), TensorError>` reads CSR `0x808` and
  returns `Ok(())` when no fault is latched. Call after `tensor_wait(Fma)` to
  confirm the FMA completed cleanly before storing results. The raw accessor
  `tensor_error()` is also available and is `#[must_use]`.
- `fence()` after `tensor_store`: guarantees that stored values reach DRAM and
  are visible to the DMA engine and other agents before the kernel returns.

### Load event IDs

Both `tensor_load` and `tensor_load_b` accept an `id: bool` parameter that
sets bit 0 of x31. This selects which `TensorWait` event the load fires:
`false` -> `Load0`, `true` -> `Load1`. The two IDs are independent, so issuing
A loads with `id: false` and B loads with `id: true` lets
`tensor_wait(Load0)` confirm A's arrival in the scratchpad without also
serialising on B's DMA (B is forward-paired with the FMA and need not be
explicitly waited for).

Only the primary hart of each Minion (`mhartid & 1 == 0`) should issue tensor
instructions. The companion hart must not touch the same L1 scratchpad lines or
FP register file concurrently.

## Scratchpad layout

Each Minion has a private 48-line L1 scratchpad (3,072 bytes; 64 bytes per
line). `tensor_load` fills lines selected by the START and ROWS fields of the
`xs` register. `tensor_store` reads from the FP register file, not from the
scratchpad.

## The `fma32_xs` bit field

`fma32_xs` constructs the 64-bit `xs` argument for `tensor_fma32` (PRM Table
9-4):

```rust,ignore
let xs = fma32_xs(
    bcols,    // B column groups minus one (0..=3; output cols = 4*(bcols+1))
    arows,    // A tile rows minus one (0..=15)
    acols,    // A tile columns minus one (0..=15); also B rows loaded by LoadB
    aoffset,  // A byte offset within each scratchpad line, in 4-byte units
    tenb,     // true = read B from TenB register file
    bstart,   // scratchpad line of B (ignored when tenb = true)
    astart,   // scratchpad line of A
    mul_only, // true = C = A*B (initialise); false = C += A*B (accumulate)
    use_mask, // true = apply tensor_mask row-enable register
);
```

For a 16x16 output tile: `bcols = 3` (4*(3+1) = 16 columns), `arows = 15`
(16 rows), `tenb = true` (B from TenB register file).

## Full kernel example: sGEMM

`et-k-rs/src/bin/sgemm.rs` is a complete, hardware-verified tensor kernel.
Its structure illustrates the canonical usage pattern:

1. Outer loop over output tiles, **shire-blocked**: each shire handles a
   contiguous `ceil(n_tiles / n_shires)` slice; within the block the 32 Minions
   distribute cyclically with step 32. This improves A-row reuse in the
   shire-shared L2 cache relative to global-cyclic assignment (+26% at N=4096).
2. Inner k-loop over the inner dimension in 16-column slices:
   - `tensor_load` A sub-tile into scratchpad lines 0..arows.
   - `tensor_wait(Load0)`.
   - `tensor_load_b` B sub-tile into TenB register file (`id: true` to use Load1, keeping A's Load0 event independent).
   - `tensor_fma32` (mul_only on first k-tile, accumulate on subsequent).
   - `tensor_wait(Fma)`.
3. `tensor_store` the accumulated C tile from FP registers f0..f31 to DRAM.
4. `fence()`.

The host launches the kernel through `et_soc1::blas::sgemm`, which validates
arguments and calls `Device::launch_spmd`. See `et-rs/examples/sgemm.rs` for
the end-to-end demonstration.

## x31 (t6) as an implicit stride register

`tensor_load`, `tensor_load_b`, and `tensor_store` each read the row stride
from `x31` (`t6`) at execution time. The `et_kernel::tensor` wrappers set `t6`
with a `mv t6, {stride}` immediately before the `CSRRW` in the same asm block,
so no separate `mv` is needed at the call site.

## PS SIMD stub

`et_kernel::simd` provides placeholder wrappers for the ET-SoC-1 packed-single
(PS) SIMD extension (PRM Chapter 5), gated on `cfg(target_feature = "f")`.
The functions compile and link but contain no real asm pending confirmation of
the PS opcode encodings. The module is marked `#[doc(hidden)]` and does not
appear in published documentation; do not depend on it in production code.
Enabling the module requires compiling with `+f` in `RUSTFLAGS` or
`.cargo/config.toml`.
