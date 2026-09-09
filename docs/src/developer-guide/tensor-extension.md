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
| `CSR_TENSOR_FMA` | `tensor_fma` | `0x801` | bits 3:1 select variant: 000=FMA32, 001=FMA16A32, 011=IMA8A32 |
| `CSR_TENSOR_REDUCE` | `tensor_reduce` | `0x800` | bits 1:0: 00=Send, 01=Recv; TensorSend/TensorRecv |
| `CSR_TENSOR_WAIT` | `tensor_wait` | `0x830` | stalls hart until the event fires |
| `CSR_TENSOR_ERROR` | `tensor_error` | `0x808` | latched co-processor error flags (`#[must_use]`) |
| `CSR_TENSOR_MASK` | `tensor_mask` | `0x805` | per-row FMA enable bits |
| `CSR_TENSOR_STORE` | `tensor_store` | `0x87F` | store from FP registers (bit 48=0) or scratchpad (bit 48=1) |
| `CSR_TENSOR_LOAD` | `tensor_load` | `0x83F` | load to scratchpad (bit 52=0) or TenB register file (bit 52=1) |
| `CSR_TENSOR_LOAD_L2` | `tensor_load_l2` | `0x85F` | prefetch to shire L2 without L1 scratchpad fill |

All constants are defined in `et_kernel::tensor` and marked `pub`.

## Concurrency model and `TensorEvent`

The tensor co-processor runs asynchronously alongside the RISC-V integer
pipeline. The hart must call `tensor_wait` with the appropriate [`TensorEvent`]
before reading results or reusing resources. `TensorEvent` is `#[non_exhaustive]`
-- always match with a catch-all arm.

| Event | Code | Fires when |
|---|---|---|
| `Load0` | 0 | `tensor_load`/`tensor_store_from_scp` with `id=false` completes |
| `Load1` | 1 | `tensor_load`/`tensor_load_b`/`tensor_store_from_scp` with `id=true` completes |
| `LoadL2_0` | 2 | `tensor_load_l2` with `id=false` completes |
| `LoadL2_1` | 3 | `tensor_load_l2` with `id=true` completes |
| `Prefetch0` | 4 | L2/L3 prefetch with `id=false` completes |
| `Prefetch1` | 5 | L2/L3 prefetch with `id=true` completes |
| `CacheOp` | 6 | L1 `evict_va`/`flush_va` cache operation completes |
| `Fma` | 7 | FMA/IMA accumulation completes |
| `Store` | 8 | `tensor_store` (from FP registers) DMA completes |
| `TensorReduce` | 9 | `tensor_send`/`tensor_recv` reduction completes |
| `TensorQuant` | 10 | quantisation pass completes |

Key ordering rules:

- `tensor_wait(TensorEvent::Load0)` before `tensor_fma32`: scratchpad lines
  filled by `tensor_load` must be visible before the FMA reads them.
- `tensor_wait(TensorEvent::Fma)` before `tensor_store`: the FP register file
  holds the final accumulated values only after Fma fires.
- `tensor_wait(TensorEvent::Store)` drains only the tensor store DMA; allows
  non-tensor scalar work to interleave. `fence()` is still required before
  returning if DMA or other agents must observe the stores.
- `check_tensor_error()` reads CSR `0x808`; call after `tensor_wait(Fma)` to
  confirm the FMA completed without fault. The raw `tensor_error()` is also
  available and is `#[must_use]`.
- `tensor_wait(TensorEvent::LoadL2_0)` / `LoadL2_1` after `tensor_load_l2`:
  the L2 prefetch fires its own pair of events (2/3), not `CacheOp`.

### Load event IDs

`tensor_load`, `tensor_load_b`, `tensor_store_from_scp`, and `tensor_load_l2`
accept an `id: bool` parameter selecting which event pair fires on completion:
`false` -> event 0 (Load0 or LoadL2_0), `true` -> event 1 (Load1 or LoadL2_1).
The two IDs are independent, allowing one load stream to be awaited without
serialising the other.

Only the primary hart of each Minion (`mhartid & 1 == 0`) should issue tensor
instructions.

## Scratchpad layout

Each Minion has a private 48-line L1 scratchpad (3,072 bytes; 64 bytes per
line). `tensor_load` fills lines selected by the START and ROWS fields of the
`xs` register. `tensor_store` reads from the FP register file, not the
scratchpad.

## FMA variants

Three FMA variants share CSR `0x801`; bits 3:1 of the `xs` argument select
which fires.

### `tensor_fma32` (bits 3:1 = 000)

Single-precision FP32 multiply-accumulate: C += A * B. Construct the `xs`
argument with `fma32_xs`:

```rust,ignore
let xs = fma32_xs(
    bcols,    // B column groups minus one (0..=3; output cols = 4*(bcols+1))
    arows,    // A tile rows minus one (0..=15)
    acols,    // A tile cols minus one (0..=15); also B rows loaded by LoadB
    aoffset,  // A byte offset in each scratchpad line, in 4-byte units
    tenb,     // true = read B from TenB register file
    bstart,   // scratchpad line of B (ignored when tenb = true)
    astart,   // scratchpad line of A
    mul_only, // true = C = A*B (initialise); false = C += A*B (accumulate)
    use_mask, // true = apply tensor_mask row-enable register
);
```

For a 16x16 output tile: `bcols=3`, `arows=15`, `tenb=true`.

### `tensor_fma16a32` (bits 3:1 = 001)

FP16 inputs accumulate into an FP32 result. Construct the `xs` argument with
`fma16a32_xs`. The ACOLS semantics differ from `fma32_xs`:

- **ACOLS = n** means K = **2*(n+1)** fp16 element pairs per clock (two
  K-columns are processed simultaneously). This is twice the depth of FMA32 for
  the same ACOLS value. Pass `acols` = K/2 - 1, **not** K - 1.
- Rounding is round-toward-zero (RTZ), not round-to-nearest. Measured RMS
  relative error against an FP32 reference is approximately 2.6e-4.
- The paired `tensor_load_b` must have `rows = acols`; hardware fires
  `tensor_error[6]` (LoadB.ROWS != ACOLS mismatch) otherwise.

### `tensor_ima8a32` (bits 3:1 = 011)

INT8 inputs accumulate into an INT32 result. Construct the `xs` argument with
`ima8a32_xs`. ACOLS semantics: **ACOLS = n** means K = **4*(n+1)** int8 elements
per clock (four elements are processed simultaneously). The DST field selects
whether the output goes to the FP register file or a TenC output register.

## `tensor_load_b` and the TenB register file

`tensor_load_b` fills the TenB register file for use by any FMA variant when
`tenb=true`. The TenB path does **not** support hardware interleaving: the B
matrix must be pre-packed host-side into the TenB layout before upload. There is
no on-device packing path.

## `tensor_store_from_scp`

`tensor_store_from_scp` reads L1 scratchpad lines directly to DRAM, bypassing
the FP register file. The `xs` field encodes the scratchpad start line, row
count, stride, and destination address. Wait for `TensorEvent::Load0` (or
`Load1` when `id=true`) before reusing the scratchpad lines.

## `tensor_send` and `tensor_recv`: inter-hart reduction

Two harts exchange FP register values and optionally combine them with a
[`ReduceFunct`] reduction:

```rust,ignore
use et_kernel::tensor::{tensor_send, tensor_recv, ReduceFunct, TensorEvent, tensor_wait};

// Sender:
unsafe { tensor_send(dest_hart, ReduceFunct::Add); }
// Receiver:
unsafe { tensor_recv(src_hart, ReduceFunct::Add); }
tensor_wait(TensorEvent::TensorReduce);
```

[`ReduceFunct`] variants: `Add`, `Max`, `Min` (signed 32-bit for Max/Min, per
PRM Table 9-8), `None` (copy without combining).

## Cache operations

`et_kernel::cache` provides L1 cache management for software-coherent cross-hart
sharing:

```rust,ignore
use et_kernel::cache::{cache_writeback, cache_invalidate, cache_flush};

// After writing shared data: flush L1 lines to DRAM so other agents see them.
cache_writeback(ptr as usize, byte_len);

// Before reading data another agent has written: invalidate stale L1 lines.
cache_invalidate(ptr as usize, byte_len);

// Flush and invalidate in one pass.
cache_flush(ptr as usize, byte_len);
```

Lower-level `_to` variants accept an explicit [`CacheDest`] to target L2 or L3
rather than DDR. All functions use `flush_va` (CSR `0x8BF`) and `evict_va`
(CSR `0x89F`) and fire `TensorEvent::CacheOp` on completion; call
`tensor_wait(TensorEvent::CacheOp)` to ensure the operation has finished.

`tensor_load_l2` (CSR `0x85F`) is a separate path: it prefetches data into the
shire's L2 cache without filling L1. It fires `TensorEvent::LoadL2_0` (id=false)
or `LoadL2_1` (id=true), not `CacheOp`.

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
   - `tensor_load_b` B sub-tile into TenB register file (`id: true` to use Load1,
     keeping A's Load0 event independent).
   - `tensor_fma32` (mul_only on first k-tile, accumulate on subsequent).
   - `tensor_wait(Fma)`.
3. `tensor_store` the accumulated C tile from FP registers f0..f31 to DRAM.
4. `fence()`.

The host launches through `et_soc1::blas::sgemm`. See `et-rs/examples/sgemm.rs`
for the end-to-end demonstration.

## x31 (t6) as an implicit stride register

`tensor_load`, `tensor_load_b`, and `tensor_store` each read the row stride
from `x31` (`t6`) at execution time. The `et_kernel::tensor` wrappers set `t6`
with a `mv t6, {stride}` immediately before the `CSRRW` in the same asm block,
so no separate `mv` is needed at the call site.

## PS SIMD stub

`et_kernel::simd` provides placeholder wrappers for the ET-SoC-1 packed-single
(PS) SIMD extension (PRM Chapter 5). The functions call `unimplemented!()` pending
confirmation of the PS opcode encodings from hardware tests. The module is
`#[doc(hidden)]` and does not appear in published documentation; do not depend on
it in production code.

[`TensorEvent`]: https://docs.rs/et-k-rs/latest/et_kernel/tensor/enum.TensorEvent.html
[`ReduceFunct`]: https://docs.rs/et-k-rs/latest/et_kernel/tensor/enum.ReduceFunct.html
[`CacheDest`]: https://docs.rs/et-k-rs/latest/et_kernel/cache/enum.CacheDest.html
