//! Shared `no_std` helpers for ET-SoC-1 compute kernels: hart identity, the
//! U-mode trace write, a hardware memory fence, and scratchpad addressing.
//!
//! This is the device-side support library for the `hello` and `spsc` kernels in
//! this package. It has no `_start` and no panic handler; each kernel binary
//! provides those.

#![no_std]

use core::arch::asm;
use core::ptr::{read_volatile, write_volatile};

/// Base of the per-hart U-mode trace control-block array
/// (`CM_UMODE_TRACE_CB_BASEADDR`); each entry is 64 bytes.
pub const CB_BASE: usize = 0x8004_F23000;
const CB_STRIDE: usize = 64;
const CB_BASE_PER_HART: usize = 24;
const CB_OFFSET_PER_HART: usize = 36;
const TRACE_TYPE_STRING: u16 = 0;
const ENTRY_HEADER_SIZE: usize = 16;
const TRACE_STRING_MAX: usize = 512;

/// Current hart ID, from the custom `hartid` CSR (`0xCD0`).
#[inline(always)]
pub fn hart_id() -> u32 {
    let v: u64;
    // SAFETY: reads a U-mode-accessible CSR with no side effects.
    unsafe { asm!("csrr {0}, 0xcd0", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v as u32
}

/// Current shire ID (`hart_id >> 6`; 64 harts per shire).
#[inline(always)]
pub fn shire_id() -> u32 {
    hart_id() >> 6
}

/// A cycle timestamp (`hpmcounter3`, CSR `0xC03`) for trace entry headers.
#[inline(always)]
pub fn timestamp() -> u64 {
    let v: u64;
    // SAFETY: reads a U-mode-accessible performance-counter CSR.
    unsafe { asm!("csrr {0}, 0xc03", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// Full hardware memory fence (`fence rw, rw`) that also bars compiler
/// reordering. This is an ordering barrier, not an atomic operation.
#[inline(always)]
pub fn fence() {
    // No `nomem`: the asm is treated as touching memory, so the compiler will
    // not move loads/stores across it either.
    unsafe { asm!("fence rw, rw", options(nostack, preserves_flags)) };
}

/// Base address of `shire`'s 2.5 MB L2 scratchpad
/// (`ETSOC_SCP_GET_SHIRE_ADDR(shire, 0)`): `0x8000_0000 | (shire << 23)`.
#[inline(always)]
pub fn scp_shire_base(shire: u32) -> usize {
    0x8000_0000usize + ((shire as usize) << 23)
}

#[inline(always)]
fn cb_index(hart: u32) -> usize {
    if hart < 2048 {
        hart as usize
    } else {
        (hart - 32) as usize
    }
}

#[inline(always)]
fn align8(n: usize) -> usize {
    (n + 7) & !7
}

/// Write `text` as a NUL-terminated string trace entry for the current hart,
/// exactly as the SDK's `Trace_String` does (reserve via the control block, then
/// write a `trace_string_t`).
pub fn trace_str(text: &[u8]) {
    let hid = hart_id();
    let str_len = align8(text.len() + 1).min(TRACE_STRING_MAX);
    let cb = CB_BASE + cb_index(hid) * CB_STRIDE;
    // SAFETY: firmware populated the CB at this fixed address before launch.
    let base = unsafe { read_volatile((cb + CB_BASE_PER_HART) as *const u64) } as usize;
    let offset = unsafe { read_volatile((cb + CB_OFFSET_PER_HART) as *const u32) };
    let head = base + offset as usize;
    // SAFETY: `head` lies within this hart's reserved trace-buffer slice.
    unsafe {
        write_volatile(head as *mut u64, timestamp());
        write_volatile((head + 8) as *mut u32, str_len as u32);
        write_volatile((head + 12) as *mut u16, hid as u16);
        write_volatile((head + 14) as *mut u16, TRACE_TYPE_STRING);
        let s = (head + ENTRY_HEADER_SIZE) as *mut u8;
        let mut i = 0;
        while i < str_len {
            let byte = if i < text.len() { text[i] } else { 0 };
            write_volatile(s.add(i), byte);
            i += 1;
        }
        write_volatile(
            (cb + CB_OFFSET_PER_HART) as *mut u32,
            offset + (ENTRY_HEADER_SIZE + str_len) as u32,
        );
    }
}

/// A fixed-capacity stack buffer for composing trace messages without `alloc`.
pub struct MsgBuf {
    buf: [u8; 192],
    len: usize,
}

impl Default for MsgBuf {
    fn default() -> Self {
        Self::new()
    }
}

impl MsgBuf {
    pub fn new() -> Self {
        MsgBuf {
            buf: [0; 192],
            len: 0,
        }
    }

    /// Append raw text (truncated if the buffer fills).
    pub fn str(&mut self, s: &[u8]) -> &mut Self {
        let mut i = 0;
        while i < s.len() && self.len < self.buf.len() {
            self.buf[self.len] = s[i];
            self.len += 1;
            i += 1;
        }
        self
    }

    /// Append a decimal integer.
    pub fn u64(&mut self, mut v: u64) -> &mut Self {
        let mut tmp = [0u8; 20];
        let mut c = 0;
        loop {
            tmp[c] = b'0' + (v % 10) as u8;
            v /= 10;
            c += 1;
            if v == 0 {
                break;
            }
        }
        while c > 0 && self.len < self.buf.len() {
            c -= 1;
            self.buf[self.len] = tmp[c];
            self.len += 1;
        }
        self
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.buf[..self.len]
    }
}

/// Cache-line size in bytes. Per-hart outputs are placed one-per-line so that
/// distinct harts never write the same line: false sharing silently corrupts
/// data on this software-coherent architecture.
pub const CACHE_LINE: usize = 64;

/// A hart's view of an SPMD launch: its identity within `n_harts` participants.
///
/// The safety story of the reduction demo lives here. A kernel body, given a
/// `Grid`, can obtain only *its own* disjoint slice of the input and *its own*
/// output cell -- it has no way to name another hart's data, so cross-hart data
/// races are unrepresentable in the (safe) kernel body. The small `unsafe`
/// boundary that turns device addresses into slices is confined to this module.
pub struct Grid {
    hart: u32,
    n_harts: u32,
}

impl Grid {
    /// Build from the current hart's id and the number of participating harts.
    pub fn new(n_harts: u32) -> Self {
        Grid {
            hart: hart_id(),
            n_harts,
        }
    }

    pub fn hart(&self) -> u32 {
        self.hart
    }

    pub fn n_harts(&self) -> u32 {
        self.n_harts
    }

    /// Whether this hart participates (the launch runs every hart of the shire,
    /// so surplus harts opt out).
    pub fn active(&self) -> bool {
        self.hart < self.n_harts
    }

    /// This hart's half-open element range of a length-`n` domain: contiguous,
    /// disjoint across harts, and together covering all of `[0, n)` (a balanced
    /// split, the first `n % n_harts` harts taking one extra element).
    fn range(&self, n: usize) -> (usize, usize) {
        let h = self.hart as usize;
        let p = (self.n_harts as usize).max(1);
        let base = n / p;
        let rem = n % p;
        let start = h * base + h.min(rem);
        let len = base + if h < rem { 1 } else { 0 };
        (start, start + len)
    }

    /// Borrow this hart's disjoint sub-slice of `data`.
    pub fn my_slice<'a, T>(&self, data: &'a [T]) -> &'a [T] {
        let (start, end) = self.range(data.len());
        &data[start..end]
    }

    /// Borrow this hart's own output cell from an array of one cache-line-padded
    /// `T` per hart based at device address `base`.
    ///
    /// # Safety
    /// `base` must address at least `n_harts * CACHE_LINE` writable bytes of
    /// device memory. Disjointness across harts is guaranteed by construction
    /// (distinct `hart` ids map to distinct cache lines).
    pub unsafe fn output_cell<'a, T>(&self, base: usize) -> &'a mut T {
        unsafe { &mut *((base + self.hart as usize * CACHE_LINE) as *mut T) }
    }
}

/// View `n` elements of type `T` at device address `addr` as a shared slice.
///
/// # Safety
/// `addr` must point to `n` valid, aligned, initialised `T` that outlive the
/// returned borrow and are not mutated through another path meanwhile.
pub unsafe fn device_slice<'a, T>(addr: usize, n: usize) -> &'a [T] {
    unsafe { core::slice::from_raw_parts(addr as *const T, n) }
}
