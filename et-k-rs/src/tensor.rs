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
//! - `TensorWait(Load0)` before `tensor_fma32`: scratchpad A is populated.
//! - `TensorWait(Fma)` before `tensor_store`: FP register file holds final C.
//! - `fence rw, rw` (via [`crate::fence`]) after `tensor_store`: stores are
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
pub const CSR_TENSOR_LOAD:  u16 = 0x83F;

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
/// after `tensor_wait` to check for co-processor faults.
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
///
/// # Safety
/// Same alignment and primary-hart constraints as [`tensor_load`].
#[inline(always)]
pub unsafe fn tensor_load_b(addr: usize, rows: u8, coop: bool, stride: u64) {
    // xs bit layout (PRM Table 9-6):
    //   63: MSK=0, 62: COOP, 61:53=0 (reserved),
    //   52=1 (TensorLoadB distinguisher),
    //   51:48=0 (reserved), 47:6=ADDR>>6, 5:4=0, 3:0=ROWS.
    let xs: u64 = ((coop as u64)  << 62)
               |  (1_u64          << 52)
               |  (addr as u64)           // 64B-aligned: bits 47:6 correct
               |  (rows as u64 & 0xF);
    unsafe {
        asm!(
            "mv t6, {stride}",
            concat!("csrrw x0, ", stringify!(0x83F), ", {xs}"),
            stride = in(reg) stride,
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
