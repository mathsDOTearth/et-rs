//! Tensor-extension intrinsics for the ET-SoC-1 Minion core.
//!
//! All tensor instructions on the ET-SoC-1 are encoded as standard RISC-V
//! `csrrw xd, <csr>, xs` writes (see PRM Chapter 9). No custom opcode or
//! target-feature extension is required: `riscv64imac` suffices because the
//! operand registers are ordinary integer GPRs (the source value `xs` is an
//! integer register; the FP register file is accessed implicitly by the
//! tensor co-processor hardware, not by the instruction encoding).
//!
//! # Concurrency model
//!
//! The tensor co-processor operates independently of the RISC-V hart's
//! integer pipeline. Issuing a tensor instruction initiates an asynchronous
//! operation; the hart must call [`tensor_wait`] with the appropriate
//! [`TensorEvent`] before reading results or reusing the scratchpad. The
//! ordering guarantees are:
//!
//! - `TensorWait(Load0)` before `tensor_fma32` / `tensor_fma16a32` /
//!   `tensor_ima8a32`: scratchpad A (and B when TENB=0) is populated.
//! - `TensorWait(Fma)` before `tensor_store` / `tensor_store_from_scp`:
//!   FP register file (or TenC for IMA8A32 with DST=0) holds final C.
//! - `TensorWait(Store)` drains only tensor store DMA; prefer it over a full
//!   `fence rw, rw` when only tensor-store ordering is required.
//! - `TensorWait(CacheOp)` after `cache_writeback` / `cache_invalidate` /
//!   `tensor_load_l2`: all cache management operations have completed.
//! - `fence rw, rw` (via [`crate::fence`]) after the final store: writes are
//!   visible to other Minions and the DMA engine before the kernel returns.
//!
//! # Scratchpad layout
//!
//! Each Minion has a private 48-line L1 scratchpad (3 072 bytes). Only the
//! primary hart of the Minion (hart 0, i.e. `mhartid & 1 == 0`) should issue
//! tensor load/store/FMA instructions; the companion hart (hart 1) must not
//! touch the same scratchpad lines concurrently.

use core::arch::asm;

// ---------------------------------------------------------------------------
// CSR addresses (PRM Chapter 9, Table 9-1)
// ---------------------------------------------------------------------------

/// TensorFMA CSR (`tensor_fma`): selects the FMA variant via xs bits 3:1.
/// (PRM Table 9-7: TensorFMA32 = 3:1 000, TensorFMA16A32 = 001, ...)
pub const CSR_TENSOR_FMA:   u16 = 0x801;
/// TensorWait CSR (`tensor_wait`): stalls the hart until the requested event.
pub const CSR_TENSOR_WAIT:  u16 = 0x830;
/// TensorError CSR (`tensor_error`): latched error flags from the co-processor.
/// (PRM Table 9-1: 0x808, not 0x831)
pub const CSR_TENSOR_ERROR: u16 = 0x808;
/// TensorMask CSR (`tensor_mask`): per-row enable bits for the A tile.
/// (PRM Table 9-1: 0x805, not 0x832)
pub const CSR_TENSOR_MASK:  u16 = 0x805;
/// TensorStore CSR (`tensor_store`): store from FP registers (bit 48 = 0) or
/// from the L1 scratchpad (bit 48 = 1 = TensorStoreFromScp) to memory.
/// (PRM Table 9-7: 0x87F, not 0x83E)
pub const CSR_TENSOR_STORE: u16 = 0x87F;
/// TensorLoad / TensorLoadB CSR (`tensor_load`): load from memory to the L1
/// scratchpad (xs bit 52 = 0) or to the TenB register file (bit 52 = 1).
pub const CSR_TENSOR_LOAD:    u16 = 0x83F;
/// TensorLoadL2Scp CSR: loads rows from memory to the shire L2 cache without
/// consuming any L1 scratchpad lines. Useful for prefetching A strips while
/// the current k-loop tile executes, so the subsequent `tensor_load` (L1 fill)
/// completes from L2 rather than DRAM.
pub const CSR_TENSOR_LOAD_L2: u16 = 0x85F;
/// TensorReduce CSR (`tensor_reduce`): hart-to-hart register-file exchange.
/// xs bits 1:0 select the variant: TensorSend=00, TensorRecv=01,
/// TensorBroadcast=10, TensorReduce=11. (PRM Table 9-7: 0x800)
pub const CSR_TENSOR_REDUCE:  u16 = 0x800;

// ---------------------------------------------------------------------------
// TensorWait event codes (PRM Table 9-2, xs bits 3:0)
// ---------------------------------------------------------------------------

/// Tensor co-processor synchronisation events for [`tensor_wait`].
///
/// The four-bit EVENT field in the TensorWait `xs` register selects which
/// outstanding operation the hart waits for before the instruction retires.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorEvent {
    /// Completion of all TensorLoad operations issued with ID = 0.
    Load0 = 0,
    /// Completion of all TensorLoad operations issued with ID = 1.
    Load1 = 1,
    /// Completion of all preceding TensorFMA operations; the FP register file
    /// holds the final accumulated C tile and may be read or stored.
    Fma   = 7,
    /// Completion of all preceding TensorStore DMA transfers (PRM Table 9-2,
    /// event code 8). Drains only the tensor store DMA, allowing the compiler
    /// more freedom to reorder non-tensor memory accesses around it. Prefer
    /// this over a full `fence rw, rw` when only tensor-store ordering is
    /// required (e.g. confirming one tile is written before reusing FP registers
    /// for the next tile in a pipelined loop).
    Store   = 8,
    /// Completion of all preceding cache management operations (EvictVA,
    /// FlushVA, PrefetchVA, TensorLoadL2Scp). Required after any cache op
    /// before issuing memory accesses to the affected cache lines (PRM
    /// Table 9-2, event code 6; PRM Section 8.1.3).
    ///
    /// Calling [`cache::cache_writeback`], [`cache::cache_invalidate`], or
    /// [`cache::cache_flush`] already issues this wait internally; use this
    /// variant directly only when batching cache ops and deferring the wait.
    CacheOp = 6,
}

// ---------------------------------------------------------------------------
// TensorError (PRM Table 9-3)
// ---------------------------------------------------------------------------

/// Tensor co-processor error status, returned by [`check_tensor_error`].
///
/// The raw value is the 64-bit content of the `tensor_error` CSR (0x808).
/// Named bit accessors will be added once PRM Table 9-3 bit positions are
/// confirmed on hardware. Use [`raw`](TensorError::raw) to inspect the value
/// directly in the interim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorError(u64);

impl TensorError {
    /// Returns the raw CSR value as read from `tensor_error` (CSR 0x808).
    #[inline]
    pub fn raw(self) -> u64 {
        self.0
    }
}

// ---------------------------------------------------------------------------
// Public intrinsic functions
// ---------------------------------------------------------------------------

/// Stall the hart until the specified tensor co-processor event fires.
///
/// This must be called between dependent tensor operations to enforce ordering
/// -- the co-processor and the hart pipeline are otherwise decoupled.
#[inline(always)]
pub fn tensor_wait(event: TensorEvent) {
    let xs: u64 = event as u64;
    // SAFETY: csrrw to a U-mode-accessible tensor CSR with no memory effects
    // from the hart's perspective; the co-processor drains its pipeline.
    unsafe {
        asm!(
            concat!("csrrw x0, ", stringify!(0x830), ", {xs}"),
            xs = in(reg) xs,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// Read the tensor co-processor error status register.
///
/// Returns 0 when no error has occurred since the last reset. A non-zero
/// value encodes the error class in bits defined by PRM Table 9-3. Call
/// after `tensor_wait` to check for co-processor faults. Prefer
/// [`check_tensor_error`] to obtain a typed result.
#[must_use = "tensor_error() returns the co-processor fault status; \
              a non-zero value indicates a hardware error that must be handled"]
#[inline(always)]
pub fn tensor_error() -> u64 {
    let v: u64;
    // SAFETY: csrrs with rs1 = x0 reads without side effect.
    unsafe {
        asm!(
            concat!("csrrs {v}, ", stringify!(0x808), ", x0"),
            v = out(reg) v,
            options(nomem, nostack, preserves_flags),
        );
    }
    v
}

/// Check the tensor co-processor error register and return a typed result.
///
/// Returns `Ok(())` when no fault has been latched. Returns `Err(TensorError)`
/// containing the raw CSR value otherwise. Call after `tensor_wait` to verify
/// that the preceding tensor operation completed without fault. Named bit
/// accessors on [`TensorError`] will be added once PRM Table 9-3 bit positions
/// are confirmed on hardware.
///
/// # Example
/// ```no_run
/// # use et_kernel::tensor::{TensorEvent, tensor_wait, check_tensor_error};
/// # unsafe {
/// tensor_wait(TensorEvent::Fma);
/// check_tensor_error().expect("TensorFMA fault");
/// # }
/// ```
#[inline(always)]
pub fn check_tensor_error() -> Result<(), TensorError> {
    let v = tensor_error();
    if v == 0 { Ok(()) } else { Err(TensorError(v)) }
}

/// Initiate an asynchronous TensorLoadL2Scp from memory into the shire L2 cache.
///
/// Identical to [`tensor_load`] in xs encoding and x31 convention, but targets
/// CSR `0x85F` (TensorLoadL2Scp) rather than `0x83F`. The rows are loaded into
/// the shire L2 without consuming any L1 scratchpad lines. Use this to prefetch
/// A strips while the current k-loop FMA executes; the subsequent
/// [`tensor_load`] for the same address will then complete from L2 rather than
/// DRAM, removing A-DMA latency from the FMA critical path.
///
/// # Parameters
/// Same as [`tensor_load`]: `addr` (64-byte aligned), `start` (L2 target line
/// index), `rows` (rows to load minus one, 0..=15), `id` (load event selector),
/// `stride` (row stride in bytes, 64-byte aligned).
///
/// # Safety
/// Same constraints as [`tensor_load`]: `addr` must be aligned and within
/// device memory; must be called from the primary hart.
#[inline(always)]
pub unsafe fn tensor_load_l2(addr: usize, start: u8, rows: u8, id: bool, stride: u64) {
    // xs layout is identical to TensorLoad; only the CSR address differs.
    let xs: u64 = ((start as u64 & 0x3F) << 53)
               |  (addr as u64)
               |  (rows as u64 & 0xF);
    unsafe {
        asm!(
            "mv t6, {stride}",
            concat!("csrrw x0, ", stringify!(0x85F), ", {xs}"),
            stride = in(reg) stride | (id as u64),  // bit 0 of x31 = ID
            xs     = in(reg) xs,
            out("t6") _,
            options(nostack),
        );
    }
}

/// Write the per-row enable mask for the next TensorFMA.
///
/// Bit `i` in `mask` enables row `i` of the A tile. Setting bit `i = 0`
/// suppresses the update to C row `i` (useful for partial M tiles when the
/// mask register is more convenient than setting AROWS). For most uses,
/// leave the mask at its reset value of all-ones and control the tile size
/// via the AROWS field in [`tensor_fma32`].
#[inline(always)]
pub fn set_tensor_mask(mask: u16) {
    let xs: u64 = mask as u64;
    unsafe {
        asm!(
            concat!("csrrw x0, ", stringify!(0x805), ", {xs}"),
            xs = in(reg) xs,
            options(nomem, nostack, preserves_flags),
        );
    }
}

/// Initiate an asynchronous TensorLoad from memory into the L1 scratchpad.
///
/// Loads `rows + 1` consecutive rows of 64 bytes each from memory into
/// L1 scratchpad lines `start` through `start + rows`. Row `i` is read from
/// address `addr + i * stride`. The operation is asynchronous: call
/// `tensor_wait(TensorEvent::Load0)` (or `Load1` if `id = true`) before
/// reading the scratchpad in a subsequent [`tensor_fma32`].
///
/// # Parameters
/// - `addr`: 64-byte aligned virtual address of the first row in memory.
/// - `start`: L1 scratchpad starting line index (0..=47).
/// - `rows`: number of rows to load minus one (ROWS field, 0..=15).
///   Loads `rows + 1` cache lines.
/// - `id`: selects the TensorWait event (false = `Load0`, true = `Load1`).
/// - `stride`: row stride in bytes (64-byte aligned); placed in x31 by this
///   function immediately before the CSRRW instruction.
///
/// # Safety
/// - `addr` must be 64-byte aligned and point to `(rows + 1) * stride` valid,
///   readable bytes of device memory.
/// - Must be called from the primary hart of the Minion (mhartid & 1 == 0).
#[inline(always)]
pub unsafe fn tensor_load(addr: usize, start: u8, rows: u8, id: bool, stride: u64) {
    // xs bit layout (PRM Table 9-5):
    //   63: MSK=0, 62: COOP=0, 61:59=000 (TensorLoad variant),
    //   58:53=START (6-bit scratchpad line index),
    //   52=0 (TensorLoad, not TensorLoadB),
    //   51:48=0 (reserved), 47:6=ADDR>>6 (addr is 64B-aligned so bits 5:0 = 0),
    //   5:4=0 (reserved), 3:0=ROWS.
    let xs: u64 = ((start as u64 & 0x3F) << 53)
               |  (addr as u64)           // bits 47:6; addr is 64B-aligned so addr & !63 == addr
               |  (rows as u64 & 0xF);
    // x31 (t6) carries the row stride; the hardware reads it implicitly.
    unsafe {
        asm!(
            "mv t6, {stride}",
            concat!("csrrw x0, ", stringify!(0x83F), ", {xs}"),
            stride = in(reg) stride | (id as u64),  // bit 0 of x31 = ID
            xs     = in(reg) xs,
            out("t6") _,
            options(nostack),
        );
    }
}

/// Initiate an asynchronous TensorLoadB from memory into the TenB register file.
///
/// Loads `rows + 1` consecutive rows of 64 bytes each from memory into the
/// dedicated TenB buffer. This forward-pairs with the next [`tensor_fma32`]
/// call that uses `tenb = true`; the FMA waits internally for the load to
/// complete, so no explicit `tensor_wait` is needed between LoadB and FMA.
///
/// # Parameters
/// - `addr`: 64-byte aligned virtual address of the first B row in memory.
/// - `rows`: B rows to load minus one (ACOLS of the subsequent FMA, 0..=15).
/// - `coop`: set for cooperative multi-hart loading (advanced; leave false).
/// - `stride`: row stride of B in bytes (64-byte aligned); placed in x31.
/// - `id`: load event identifier placed in bit 0 of x31 (false = `Load0`,
///   true = `Load1`). Use `Load1` when a `tensor_load` with `id: false` is
///   also in flight, so that `tensor_wait(Load0)` waits only for the A tile
///   and not for the B DMA (which forward-pairs with the FMA anyway).
///
/// # Safety
/// Same alignment and primary-hart constraints as [`tensor_load`].
#[inline(always)]
pub unsafe fn tensor_load_b(addr: usize, rows: u8, coop: bool, stride: u64, id: bool) {
    // xs bit layout (PRM Table 9-6):
    //   63: MSK=0, 62: COOP, 61:53=0 (reserved),
    //   52=1 (TensorLoadB distinguisher),
    //   51:48=0 (reserved), 47:6=ADDR>>6, 5:4=0, 3:0=ROWS.
    // x31 bit 0 = ID (identical mechanism to TensorLoad; PRM Chapter 9).
    let xs: u64 = ((coop as u64)  << 62)
               |  (1_u64          << 52)
               |  (addr as u64)           // 64B-aligned: bits 47:6 correct
               |  (rows as u64 & 0xF);
    unsafe {
        asm!(
            "mv t6, {stride}",
            concat!("csrrw x0, ", stringify!(0x83F), ", {xs}"),
            stride = in(reg) stride | (id as u64),  // bit 0 of x31 = ID
            xs     = in(reg) xs,
            out("t6") _,
            options(nostack),
        );
    }
}

/// Build the xs value for a TensorFMA32 instruction.
///
/// The FMA computes C[i][j] += A[i][k] * B[k][j] (or C = A*B when
/// `mul_only = true`), accumulating into the FP register file.
///
/// # Parameters
/// - `bcols`:    B column groups minus one (BCOLS field, 0..=3; output columns
///   = 4*(bcols+1), e.g. 3 -> 16 columns).
/// - `arows`:    A tile rows minus one (AROWS field, 0..=15).
/// - `acols`:    A tile columns minus one (ACOLS field, 0..=15); also the
///   number of B rows loaded by the preceding [`tensor_load_b`].
/// - `aoffset`:  byte offset within each scratchpad line where A row data
///   begins, in 4-byte units (AOFFSET, 0..=15). Use 0 when A columns start
///   at the beginning of a cache line.
/// - `tenb`:     `true` to read B from the TenB register file (filled by the
///   preceding [`tensor_load_b`]); `false` to read from the L1 scratchpad
///   at `bstart`.
/// - `bstart`:   scratchpad line index of B (ignored when `tenb = true`).
/// - `astart`:   scratchpad line index of A (ASTART field, 0..=47).
/// - `mul_only`: `true` for C = A*B (ignore existing FP register values);
///   `false` for C += A*B (accumulate into current FP registers).
/// - `use_mask`: apply the tensor_mask row-enable register.
#[must_use = "the returned xs value must be passed to tensor_fma32; discarding it issues no instruction"]
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn fma32_xs(
    bcols:    u8,
    arows:    u8,
    acols:    u8,
    aoffset:  u8,
    tenb:     bool,
    bstart:   u8,
    astart:   u8,
    mul_only: bool,
    use_mask: bool,
) -> u64 {
    // xs bit layout (PRM Table 9-4):
    //   63: MSK, 62:57: reserved (0), 56:55: BCOLS, 54:51: AROWS,
    //   50:47: ACOLS, 46:43: AOFFSET, 42:21: reserved (0), 20: TENB,
    //   19:18: reserved (0), 17:12: BSTART, 11:10: reserved (0),
    //   9:4: ASTART, 3:1: 000 (FMA32 TensorType), 0: MUL.
    ((use_mask as u64)       << 63)
  | ((bcols   as u64 & 0x3)  << 55)
  | ((arows   as u64 & 0xF)  << 51)
  | ((acols   as u64 & 0xF)  << 47)
  | ((aoffset as u64 & 0xF)  << 43)
  | ((tenb    as u64)        << 20)
  | ((bstart  as u64 & 0x3F) << 12)
  | ((astart  as u64 & 0x3F) <<  4)
  // bits 3:1 = 000 (FMA32 TensorType selector)
  | (mul_only as u64)
}

/// Initiate an asynchronous TensorFMA32.
///
/// Issues `csrrw x0, 0x801, xs` where `xs` is built by [`fma32_xs`]. The
/// operation is asynchronous: call `tensor_wait(TensorEvent::Fma)` before
/// reading the FP register file or issuing a subsequent [`tensor_store`].
///
/// # Safety
/// - The L1 scratchpad must be fully populated (TensorLoad with subsequent
///   `tensor_wait(Load0)`) before this call when `tenb = false`, or
///   equivalently [`tensor_load_b`] must have been issued before this call
///   for the TenB path.
/// - Must be called from the primary hart of the Minion.
#[inline(always)]
pub unsafe fn tensor_fma32(xs: u64) {
    unsafe {
        asm!(
            concat!("csrrw x0, ", stringify!(0x801), ", {xs}"),
            xs = in(reg) xs,
            options(nostack),
        );
    }
}

/// Build the xs value for a TensorFMA16A32 instruction.
///
/// Computes C[i][j] += A[i][k]*B[k][j] + A[i][k+1]*B[k+1][j] (with an fused
/// 3-way addition that is not IEEE754-equivalent to two separate adds), where A
/// and B contain fp16 elements; C accumulates as fp32. The xs layout is
/// identical to [`fma32_xs`] except bits 3:1 = `001` (FMA16A32 TensorType).
///
/// # Parameters
/// (identical to [`fma32_xs`]; `tenb` true selects the TenB register file for B.)
#[must_use = "the returned xs value must be passed to tensor_fma16a32; \
              discarding it issues no instruction"]
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn fma16a32_xs(
    bcols:    u8,
    arows:    u8,
    acols:    u8,
    aoffset:  u8,
    tenb:     bool,
    bstart:   u8,
    astart:   u8,
    mul_only: bool,
    use_mask: bool,
) -> u64 {
    // xs bit layout (PRM Table 9-4, TensorFMA16A32 variant):
    //   identical to TensorFMA32 (fma32_xs) except bits 3:1 = 001.
    ((use_mask as u64)       << 63)
  | ((bcols   as u64 & 0x3)  << 55)
  | ((arows   as u64 & 0xF)  << 51)
  | ((acols   as u64 & 0xF)  << 47)
  | ((aoffset as u64 & 0xF)  << 43)
  | ((tenb    as u64)        << 20)
  | ((bstart  as u64 & 0x3F) << 12)
  | ((astart  as u64 & 0x3F) <<  4)
  | (1_u64                    <<  1)  // bits 3:1 = 001 (FMA16A32 TensorType)
  | (mul_only as u64)
}

/// Initiate an asynchronous TensorFMA16A32.
///
/// Issues `csrrw x0, 0x801, xs` where `xs` is built by [`fma16a32_xs`].
/// The hardware selects the FMA16A32 path via bits 3:1 = `001` in xs.
/// Call `tensor_wait(TensorEvent::Fma)` before reading results.
///
/// # Safety
/// Same constraints as [`tensor_fma32`].
#[inline(always)]
pub unsafe fn tensor_fma16a32(xs: u64) {
    unsafe {
        asm!(
            concat!("csrrw x0, ", stringify!(0x801), ", {xs}"),
            xs = in(reg) xs,
            options(nostack),
        );
    }
}

/// Build the xs value for a TensorIMA8A32 instruction.
///
/// Computes C[i][j] += A[i][k] * B[k][j] (or C = A*B when `mul_only = true`),
/// where A and B hold 8-bit integer elements and C accumulates as 32-bit signed
/// integers. The A matrix is `(AROWS+1) x (ACOLS+1)*4` int8 elements; the B
/// matrix is `(ACOLS+1)*4 x (BCOLS+1)*16` int8 elements (interleaved 4 columns
/// at a time); the output is `(AROWS+1) x (BCOLS+1)*4` int32 values.
///
/// # Parameters
/// - `bcols`:      B column groups minus one (BCOLS, 0..=3; output columns = 4*(bcols+1)).
/// - `arows`:      A tile rows minus one (AROWS, 0..=15).
/// - `acols`:      A tile columns minus one (ACOLS, 0..=15).
/// - `aoffset`:    Byte offset within each scratchpad line for A data, in 4-byte units
///   (AOFFSET, 0..=15).
/// - `b_in_mem`:   `true` if B is transferred via the memory DMA path; `false` for L1
///   scratchpad. (TENB = 1 means memory for IMA8A32, unlike FMA where TENB=1 is TenB
///   register file.)
/// - `bstart`:     Starting scratchpad line for B; ignored when `b_in_mem = true`.
/// - `astart`:     Starting scratchpad line for A (ASTART, 0..=47).
/// - `dst_fp`:     `true` to write the int32 result to the FP register file;
///   `false` to write to the TenC register file. (DST, xs bit 23)
/// - `b_unsigned`: `true` if B elements are unsigned; `false` for signed.
/// - `a_unsigned`: `true` if A elements are unsigned; `false` for signed.
/// - `mul_only`:   `true` for C = A*B; `false` for C += A*B.
/// - `use_mask`:   Apply the tensor_mask row-enable register.
#[must_use = "the returned xs value must be passed to tensor_ima8a32; \
              discarding it issues no instruction"]
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn ima8a32_xs(
    bcols:      u8,
    arows:      u8,
    acols:      u8,
    aoffset:    u8,
    b_in_mem:   bool,
    bstart:     u8,
    astart:     u8,
    dst_fp:     bool,
    b_unsigned: bool,
    a_unsigned: bool,
    mul_only:   bool,
    use_mask:   bool,
) -> u64 {
    // xs bit layout (PRM Table 9-4, TensorIMA8A32 variant):
    //   63: MSK, 62:57: reserved (0), 56:55: BCOLS, 54:51: AROWS,
    //   50:47: ACOLS, 46:43: AOFFSET, 42:24: reserved (0),
    //   23: DST (0=TenC, 1=FP registers), 22: UB, 21: UA,
    //   20: TENB (0=L1 scratchpad, 1=memory path for IMA8A32),
    //   19:18: reserved (0), 17:12: BSTART, 11:10: reserved (0),
    //   9:4: ASTART, 3:1: 011 (IMA8A32 TensorType), 0: MUL.
    ((use_mask   as u64)        << 63)
  | ((bcols      as u64 & 0x3)  << 55)
  | ((arows      as u64 & 0xF)  << 51)
  | ((acols      as u64 & 0xF)  << 47)
  | ((aoffset    as u64 & 0xF)  << 43)
  | ((dst_fp     as u64)        << 23)
  | ((b_unsigned as u64)        << 22)
  | ((a_unsigned as u64)        << 21)
  | ((b_in_mem   as u64)        << 20)
  | ((bstart     as u64 & 0x3F) << 12)
  | ((astart     as u64 & 0x3F) <<  4)
  | (3_u64                       <<  1)  // bits 3:1 = 011 (IMA8A32 TensorType)
  | (mul_only    as u64)
}

/// Initiate an asynchronous TensorIMA8A32.
///
/// Issues `csrrw x0, 0x801, xs` where `xs` is built by [`ima8a32_xs`].
/// The hardware selects the integer-GEMM path via bits 3:1 = `011` in xs.
/// Call `tensor_wait(TensorEvent::Fma)` before reading results.
///
/// # Safety
/// Same constraints as [`tensor_fma32`].
#[inline(always)]
pub unsafe fn tensor_ima8a32(xs: u64) {
    unsafe {
        asm!(
            concat!("csrrw x0, ", stringify!(0x801), ", {xs}"),
            xs = in(reg) xs,
            options(nostack),
        );
    }
}

/// Initiate an asynchronous TensorStoreFromScp to memory from the L1 scratchpad.
///
/// Stores `rows + 1` 64-byte scratchpad lines to memory, bypassing the L1 data
/// cache and L2 cache. Consecutive scratchpad lines are spaced `step` lines apart
/// (so `step = 1` stores consecutive lines); consecutive destination rows are
/// spaced `stride` bytes apart.
///
/// # Parameters
/// - `addr`:   64-byte aligned virtual address of the first destination row.
/// - `rows`:   Number of rows to store minus one (ROWS, 0..=15).
/// - `start`:  Starting L1 scratchpad cache line (0..=47).
/// - `step`:   Scratchpad line stride (1..=4); encoded as STEP = step - 1.
/// - `stride`: Destination row stride in bytes; placed in x31. Low 6 bits are
///   ignored by the hardware (rows are 64-byte aligned in memory).
///
/// # Safety
/// - `addr` must be 64-byte aligned and point to `(rows + 1) * stride` bytes of
///   writable device memory.
/// - Must be called from the primary hart of the Minion.
#[inline(always)]
pub unsafe fn tensor_store_from_scp(addr: usize, rows: u8, start: u8, step: u8, stride: u64) {
    // xs bit layout (PRM Table 9-7, TensorStoreFromScp):
    //   63:62: STEP (step-1; scratchpad line stride), 61:56: START (first
    //   scratchpad line), 55: reserved (0), 54:51: ROWS (rows-1), 50:49:
    //   reserved (0), 48: 1 (TensorStoreFromScp discriminator, bit 48=0 is
    //   TensorStore), 47:6: ADDR (virtual address with 6 low-order bits
    //   omitted; addr is 64B-aligned), 5:0: reserved (0).
    // x31 carries the destination row stride; low 6 bits are ignored by hardware.
    let xs: u64 = (((step as u64).saturating_sub(1) & 0x3) << 62)
               |  ((start as u64 & 0x3F) << 56)
               |  ((rows  as u64 & 0xF)  << 51)
               |  (1_u64                 << 48)         // source = L1 scratchpad
               |  (addr as u64 & 0x0000_FFFF_FFFF_FFC0_usize as u64);  // ADDR[47:6]
    unsafe {
        asm!(
            "mv t6, {stride}",
            concat!("csrrw x0, ", stringify!(0x87F), ", {xs}"),
            stride = in(reg) stride,
            xs     = in(reg) xs,
            out("t6") _,
            options(nostack),
        );
    }
}

/// Reduction function selector for [`tensor_recv`].
///
/// Specifies how the received values are combined with the values already held
/// in the destination FP registers. (PRM Table 9-8, FUNCT field.)
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceFunct {
    /// C[i] = C[i] + src[i]  (fp32 addition)
    Fadd  = 0,
    /// C[i] = fmax(C[i], src[i])
    Fmax  = 2,
    /// C[i] = fmin(C[i], src[i])
    Fmin  = 3,
    /// C[i] = C[i] + src[i]  (integer addition on bit pattern)
    Add   = 4,
    /// C[i] = max(C[i], src[i])  (unsigned integer comparison)
    Max   = 6,
    /// C[i] = min(C[i], src[i])  (unsigned integer comparison)
    Min   = 7,
    /// C[i] = src[i]             (unconditional move)
    Move  = 8,
}

/// Initiate an asynchronous TensorSend.
///
/// Pushes `count` consecutive FP registers starting at `freg` from this hart
/// to hart 0 of the Minion identified by `target`. The partner hart must issue
/// a matching [`tensor_recv`]. This is the low-level primitive for hart-to-hart
/// reduction without software memory traffic.
///
/// # Parameters
/// - `freg`:   Starting FP register index (0..=31).
/// - `count`:  Number of FP registers to send (COUNT field, 0..=127).
/// - `target`: Destination Minion ID (TARGET field, bits 15:3 of xs).
///
/// # Safety
/// - The partner hart must call [`tensor_recv`] with the matching `source` and
///   `count` before the send retires.
/// - Must be called from the primary hart of the Minion.
#[inline(always)]
pub unsafe fn tensor_send(freg: u8, count: u8, target: u16) {
    // xs bit layout (PRM Table 9-8, TensorSend):
    //   63:62: reserved (0), 61:57: FREG (starting FP register),
    //   56:23: reserved (0), 22:16: COUNT (number of registers),
    //   15:3: TARGET (destination Minion ID), 2: reserved (0), 1:0: 00.
    let xs: u64 = ((freg   as u64 & 0x1F)  << 57)
               |  ((count  as u64 & 0x7F)  << 16)
               |  ((target as u64 & 0x1FFF) << 3);
    // bits 1:0 = 00 (TensorSend) -- naturally zero.
    unsafe {
        asm!(
            concat!("csrrw x0, ", stringify!(0x800), ", {xs}"),
            xs = in(reg) xs,
            options(nostack),
        );
    }
}

/// Initiate an asynchronous TensorRecv.
///
/// Receives `count` FP registers from the Minion identified by `source` and
/// combines them with the local FP registers starting at `freg` using the
/// operation specified by `funct`. This is the matching receive primitive for
/// [`tensor_send`].
///
/// # Parameters
/// - `freg`:   Starting local FP register index (0..=31).
/// - `funct`:  Combination operation applied to received and local values.
/// - `count`:  Number of FP registers to receive (0..=127); must match the
///   sender's `count`.
/// - `source`: Source Minion ID (SOURCE field, bits 15:3 of xs).
///
/// # Safety
/// - The partner hart must have called [`tensor_send`] before this retires.
/// - Must be called from the primary hart of the Minion.
#[inline(always)]
pub unsafe fn tensor_recv(freg: u8, funct: ReduceFunct, count: u8, source: u16) {
    // xs bit layout (PRM Table 9-8, TensorRecv):
    //   63:62: reserved (0), 61:57: FREG, 27:24: FUNCT, 23: reserved (0),
    //   22:16: COUNT, 15:3: SOURCE, 2: reserved (0), 1:0: 01 (TensorRecv).
    let xs: u64 = ((freg   as u64 & 0x1F)  << 57)
               |  ((funct  as u64 & 0xF)   << 24)
               |  ((count  as u64 & 0x7F)  << 16)
               |  ((source as u64 & 0x1FFF) << 3)
               |  1_u64;  // bits 1:0 = 01 (TensorRecv)
    unsafe {
        asm!(
            concat!("csrrw x0, ", stringify!(0x800), ", {xs}"),
            xs = in(reg) xs,
            options(nostack),
        );
    }
}

/// Initiate an asynchronous TensorStore from the FP register file to memory.
///
/// Stores `arows + 1` rows of 64 bytes each (16 f32 per row, occupying two
/// consecutive 256-bit FP registers) to memory. Row `i` is stored to
/// address `addr + i * stride`, reading from FP registers f[2i] and f[2i+1].
/// The operation is asynchronous: call [`crate::fence`] after to guarantee
/// visibility to other agents before the kernel returns.
///
/// # Parameters
/// - `addr`:  64-byte aligned virtual address of the first C row in memory.
/// - `arows`: number of C rows to store minus one (ROWS field, 0..=15).
/// - `stride`: row stride of C in bytes (64-byte aligned); placed in x31.
///
/// # Safety
/// - `addr` must be 64-byte aligned and point to `(arows + 1) * stride` bytes
///   of writable device memory.
/// - `tensor_wait(TensorEvent::Fma)` must have been called first.
/// - Must be called from the primary hart of the Minion.
#[inline(always)]
pub unsafe fn tensor_store(addr: usize, arows: u8, stride: u64) {
    // xs bit layout (PRM Table 9-7):
    //   63:62: STEP=0 (fstep=1; row i uses f[2i] and f[2i+1]),
    //   61:57: FREG=0 (start at f0),
    //   56:55: SIZE=3 (64 bytes = 16 f32 per row, two 256-bit registers),
    //   54:51: ROWS,
    //   50:49: COOP=0 (no cooperative multi-hart store),
    //   48:   0 (store from FP registers, not from Scp),
    //   47:4: ADDR >> 4 (addr is 64B-aligned, so addr & !0xF == addr),
    //   3:0:  0000.
    // Zero fields (STEP=0 at 63:62, FREG=0 at 61:57, COOP=0 at 50:49,
    // source=FP-registers at 48) are left as the natural zero of u64.
    let xs: u64 = (3_u64 << 55)                        // SIZE=3 (64B/row)
               |  ((arows as u64) << 51)               // ROWS
               |  (addr as u64 & !0xF_usize as u64);   // ADDR[47:4]; addr is 64B-aligned
    // x31 carries the C row stride; TensorStore uses bits [47:4] of x31.
    unsafe {
        asm!(
            "mv t6, {stride}",
            concat!("csrrw x0, ", stringify!(0x87F), ", {xs}"),
            stride = in(reg) stride,
            xs     = in(reg) xs,
            out("t6") _,
            options(nostack),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify the FMA32 xs bit packing for a standard full-tile configuration:
    /// BCOLS=3, AROWS=15, ACOLS=15, AOFFSET=0, TENB=1, BSTART=0, ASTART=0.
    #[test]
    fn fma32_xs_full_tile() {
        let xs = fma32_xs(3, 15, 15, 0, true, 0, 0, false, false);
        // BCOLS=3 at bits 56:55 -> 3 << 55
        assert_eq!(xs & (0x3 << 55), 3 << 55);
        // AROWS=15 at bits 54:51 -> 15 << 51
        assert_eq!(xs & (0xF << 51), 15 << 51);
        // ACOLS=15 at bits 50:47
        assert_eq!(xs & (0xF << 47), 15 << 47);
        // TENB=1 at bit 20
        assert_eq!(xs & (1 << 20), 1 << 20);
        // MUL=0, MSK=0
        assert_eq!(xs & 1, 0);
        assert_eq!(xs >> 63, 0);
    }

    /// Verify mul_only sets bit 0.
    #[test]
    fn fma32_xs_mul_only() {
        let xs = fma32_xs(3, 15, 15, 0, true, 0, 0, true, false);
        assert_eq!(xs & 1, 1);
    }

    /// Verify that TensorEvent discriminants match PRM Table 9-2.
    #[test]
    fn tensor_event_discriminants() {
        assert_eq!(TensorEvent::Load0 as u64, 0);
        assert_eq!(TensorEvent::Load1 as u64, 1);
        assert_eq!(TensorEvent::Fma   as u64, 7);
        assert_eq!(TensorEvent::Store as u64, 8);
    }

    /// Verify TensorLoad xs encoding for addr=0x1000, start=0, rows=15.
    #[test]
    fn tensor_load_xs_encoding() {
        let addr: usize = 0x0080_0000_1000; // 64B-aligned
        let start: u8 = 0;
        let rows: u8 = 15;
        let xs: u64 = ((start as u64 & 0x3F) << 53)
                   |  (addr as u64)
                   |  (rows as u64 & 0xF);
        // START field (bits 58:53) = 0
        assert_eq!((xs >> 53) & 0x3F, 0);
        // bit 52 = 0 (TensorLoad, not TensorLoadB)
        assert_eq!((xs >> 52) & 1, 0);
        // ROWS = 15
        assert_eq!(xs & 0xF, 15);
        // ADDR embedded at bits 47:6 (addr = 0x80_0000_1000, bits fit in 47:6)
        let addr_bits = addr as u64 & 0x0000_FFFF_FFFF_FFFF;
        assert_eq!(xs & addr_bits, addr_bits);
    }

    /// Verify that fma16a32_xs differs from fma32_xs only in bits 3:1.
    #[test]
    fn fma16a32_xs_tensortype() {
        let xs32  = fma32_xs(3, 15, 15, 0, true, 0, 0, false, false);
        let xs16  = fma16a32_xs(3, 15, 15, 0, true, 0, 0, false, false);
        // bits 3:1 must be 001 (value 2) for FMA16A32
        assert_eq!((xs16 >> 1) & 0x7, 1);
        // all other bits identical
        assert_eq!(xs32 & !(0x7 << 1), xs16 & !(0x7 << 1));
    }

    /// Verify ima8a32_xs bits 3:1 = 011 and the DST/UA/UB fields.
    #[test]
    fn ima8a32_xs_fields() {
        let xs = ima8a32_xs(
            /*bcols*/      3,
            /*arows*/     15,
            /*acols*/     15,
            /*aoffset*/    0,
            /*b_in_mem*/ false,
            /*bstart*/     0,
            /*astart*/     0,
            /*dst_fp*/  true,
            /*b_unsigned*/ true,
            /*a_unsigned*/ true,
            /*mul_only*/ false,
            /*use_mask*/ false,
        );
        // TensorType bits 3:1 = 011
        assert_eq!((xs >> 1) & 0x7, 3);
        // DST = 1 at bit 23
        assert_eq!((xs >> 23) & 1, 1);
        // UB = 1 at bit 22
        assert_eq!((xs >> 22) & 1, 1);
        // UA = 1 at bit 21
        assert_eq!((xs >> 21) & 1, 1);
        // TENB = 0 (b_in_mem = false)
        assert_eq!((xs >> 20) & 1, 0);
        // BCOLS, AROWS, ACOLS
        assert_eq!((xs >> 55) & 0x3, 3);
        assert_eq!((xs >> 51) & 0xF, 15);
        assert_eq!((xs >> 47) & 0xF, 15);
    }

    /// Verify ima8a32_xs with b_in_mem=true sets TENB bit.
    #[test]
    fn ima8a32_xs_b_in_mem() {
        let xs = ima8a32_xs(0, 0, 0, 0, true, 0, 0, false, false, false, false, false);
        assert_eq!((xs >> 20) & 1, 1);  // TENB = 1 (memory path)
    }

    /// Verify tensor_store_from_scp xs: bit 48 = 1, STEP, START, ROWS, ADDR.
    #[test]
    fn store_from_scp_xs_fields() {
        let addr: usize = 0x0080_0000_2000;  // 64B-aligned
        let xs: u64 = (((4_u64 - 1) & 0x3) << 62)  // step=4 -> STEP=3
                   |  ((12_u64 & 0x3F) << 56)        // start=12
                   |  ((7_u64  & 0xF)  << 51)        // rows=7
                   |  (1_u64           << 48)         // source = scratchpad
                   |  (addr as u64 & 0x0000_FFFF_FFFF_FFC0_usize as u64);
        // bit 48 = 1 (TensorStoreFromScp discriminator)
        assert_eq!((xs >> 48) & 1, 1);
        // STEP = 3 (step - 1) at bits 63:62
        assert_eq!(xs >> 62, 3);
        // START = 12 at bits 61:56
        assert_eq!((xs >> 56) & 0x3F, 12);
        // ROWS = 7 at bits 54:51
        assert_eq!((xs >> 51) & 0xF, 7);
        // ADDR embedded at bits 47:6 (addr is 64B-aligned)
        assert_eq!(xs & addr as u64, addr as u64);
    }

    /// Verify tensor_send xs: bits 1:0 = 00, FREG, COUNT, TARGET fields.
    #[test]
    fn tensor_send_xs_fields() {
        let freg: u8 = 16;
        let count: u8 = 8;
        let target: u16 = 5;
        let xs: u64 = ((freg   as u64 & 0x1F)   << 57)
                   |  ((count  as u64 & 0x7F)   << 16)
                   |  ((target as u64 & 0x1FFF) << 3);
        // bits 1:0 = 00 (TensorSend)
        assert_eq!(xs & 0x3, 0);
        // FREG at bits 61:57
        assert_eq!((xs >> 57) & 0x1F, 16);
        // COUNT at bits 22:16
        assert_eq!((xs >> 16) & 0x7F, 8);
        // TARGET at bits 15:3
        assert_eq!((xs >> 3) & 0x1FFF, 5);
    }

    /// Verify tensor_recv xs: bits 1:0 = 01, FUNCT field.
    #[test]
    fn tensor_recv_xs_fields() {
        let xs: u64 = ((4_u64 & 0x1F)   << 57)   // freg=4
                   |  ((ReduceFunct::Fadd as u64 & 0xF) << 24)  // FUNCT=0 (FADD)
                   |  ((16_u64 & 0x7F)  << 16)   // count=16
                   |  ((3_u64 & 0x1FFF) << 3)    // source=3
                   |  1_u64;                      // bits 1:0 = 01 (TensorRecv)
        // bits 1:0 = 01
        assert_eq!(xs & 0x3, 1);
        // FUNCT = 0 (FADD) at bits 27:24
        assert_eq!((xs >> 24) & 0xF, 0);
        // FREG at bits 61:57
        assert_eq!((xs >> 57) & 0x1F, 4);
    }

    /// Verify TensorStore xs encoding: STEP=0, FREG=0, SIZE=3.
    #[test]
    fn tensor_store_xs_fields() {
        let addr: usize = 0x0080_0000_2000; // 64B-aligned
        let arows: u8 = 7;
        let xs: u64 = (3_u64 << 55)              // SIZE=3
                   |  ((arows as u64) << 51)
                   |  (addr as u64 & !0xF_usize as u64);
        // SIZE = 3 at bits 56:55
        assert_eq!((xs >> 55) & 0x3, 3);
        // ROWS = 7 at bits 54:51
        assert_eq!((xs >> 51) & 0xF, 7);
        // STEP = 0 at bits 63:62
        assert_eq!(xs >> 62, 0);
        // FREG = 0 at bits 61:57
        assert_eq!((xs >> 57) & 0x1F, 0);
        // ADDR is embedded (addr is 64B-aligned, so !0xF == addr)
        assert_eq!(xs & (addr as u64), addr as u64);
    }
}
