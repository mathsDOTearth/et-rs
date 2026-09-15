//! Packed-single (PS) SIMD intrinsics for the ET-SoC-1 Minion FP register file.
//!
//! The ET-SoC-1 PS extension operates on the 256-bit FP registers (f0..f31),
//! treating each as a vector of eight single-precision (f32) lanes. PS
//! instructions share the standard RISC-V FP register file and therefore
//! require the `f` target feature to be enabled at compile time.
//!
//! # Enabling this module
//!
//! The module is gated on `cfg(target_feature = "f")`. Add `+f` via
//! `RUSTFLAGS` or `.cargo/config.toml`:
//!
//! ```toml
//! [target.riscv64imac-unknown-none-elf]
//! rustflags = ["-C", "target-feature=+f"]
//! ```
//!
//! Without `+f` the module is empty; callers must be similarly gated.
//!
//! # Register assignment for a 16-row C tile
//!
//! A 16-row GEMM C tile occupies the full FP register file (f0..f31) in
//! register pairs: row N maps to (f[2N], f[2N+1]). The broadcast scratch
//! register for [`broadcast_ps`] and [`fmul_ps_row`] must not overlap with
//! the pair being scaled.
//!
//! For tiles of at most 14 rows, pass [`PS_SCRATCH_DEFAULT`] (f28) as
//! `scratch`. For a full 16-row tile, rows 14 (f28/f29) and 15 (f30/f31)
//! conflict with f28, so the caller must choose a different scratch register:
//!
//! 1. Spill the chosen scratch register to the stack (one FP register via
//!    `fsw`/`fld` with a suitable frame slot).
//! 2. Call [`broadcast_ps`]`(alpha, scratch)`.
//! 3. Call [`fmul_ps_row`] for the conflicting row.
//! 4. Restore the scratch register from the stack.
//!
//! # Encodings
//!
//! Source: `gdb/include/opcode/esperanto-opc.h` in the ET-SoC-1 binutils fork.
//! All PS arithmetic instructions use the RISC-V custom-3 opcode space (0x7b).
//!
//! | Instruction  | MATCH          | Format | Notes |
//! |---|---|---|---|
//! | `fmul.ps`    | `0x1000_007b`  | R-type | fd = fs1 .* fs2 (element-wise). funct7=0x08, funct3=0. |
//! | `fbc.ps`     | `0x0000_000b`  | I-type | Load 4 B from `rs1+imm`, broadcast to all 8 lanes of fd. opcode=0x0b, funct3=0. |
//! | `fbcx.ps`    | `0x0000_300b`  | I-type | Broadcast GPR `rs1` to all 8 lanes of fd; imm=0 fixed. opcode=0x0b, funct3=3. |
//! | `fmvs.x.ps`  | `0xe000_207b`  | R-type | Extract lane `rs2[2:0]` from PS register `rs1` to GPR `rd`. funct7=0x70, funct3=2. |

#[cfg(target_arch = "riscv64")]
#[cfg(target_feature = "f")]
mod inner {
    use core::arch::asm;

    /// Default PS broadcast scratch register (f28 / ft8).
    ///
    /// Safe for C tiles with at most 14 rows (rows 0..=13). Rows 14 and 15
    /// place C-tile data in f28..f31, conflicting with this value; see the
    /// module doc for the spill/restore pattern required for full 16-row tiles.
    pub const PS_SCRATCH_DEFAULT: u8 = 28;

    // Emit two FMUL.PS instructions for a register pair with literal operands.
    // R-type: opcode=0x7b, funct3=0, funct7=8.
    // fd = fs1 = f{lo} or f{hi}; fs2 = f{s}.
    macro_rules! fmul2 {
        ($lo:literal, $hi:literal, $s:literal) => {
            asm!(
                concat!(
                    ".insn r 0x7b, 0, 8, f", $lo, ", f", $lo, ", f", $s, "\n",
                    ".insn r 0x7b, 0, 8, f", $hi, ", f", $hi, ", f", $s
                ),
                options(nostack, preserves_flags)
            )
        };
    }

    // Dispatch FMUL.PS for register pair ($lo, $hi) over all 32 scratch registers.
    macro_rules! scale_row {
        (($lo:literal, $hi:literal), $s:expr) => {
            match $s {
                0  => { fmul2!($lo, $hi, 0)  },
                1  => { fmul2!($lo, $hi, 1)  },
                2  => { fmul2!($lo, $hi, 2)  },
                3  => { fmul2!($lo, $hi, 3)  },
                4  => { fmul2!($lo, $hi, 4)  },
                5  => { fmul2!($lo, $hi, 5)  },
                6  => { fmul2!($lo, $hi, 6)  },
                7  => { fmul2!($lo, $hi, 7)  },
                8  => { fmul2!($lo, $hi, 8)  },
                9  => { fmul2!($lo, $hi, 9)  },
                10 => { fmul2!($lo, $hi, 10) },
                11 => { fmul2!($lo, $hi, 11) },
                12 => { fmul2!($lo, $hi, 12) },
                13 => { fmul2!($lo, $hi, 13) },
                14 => { fmul2!($lo, $hi, 14) },
                15 => { fmul2!($lo, $hi, 15) },
                16 => { fmul2!($lo, $hi, 16) },
                17 => { fmul2!($lo, $hi, 17) },
                18 => { fmul2!($lo, $hi, 18) },
                19 => { fmul2!($lo, $hi, 19) },
                20 => { fmul2!($lo, $hi, 20) },
                21 => { fmul2!($lo, $hi, 21) },
                22 => { fmul2!($lo, $hi, 22) },
                23 => { fmul2!($lo, $hi, 23) },
                24 => { fmul2!($lo, $hi, 24) },
                25 => { fmul2!($lo, $hi, 25) },
                26 => { fmul2!($lo, $hi, 26) },
                27 => { fmul2!($lo, $hi, 27) },
                28 => { fmul2!($lo, $hi, 28) },
                29 => { fmul2!($lo, $hi, 29) },
                30 => { fmul2!($lo, $hi, 30) },
                31 => { fmul2!($lo, $hi, 31) },
                _  => {}
            }
        };
    }

    /// Broadcast `scalar` to all eight lanes of PS register `dest` (0..=31).
    ///
    /// Issues `fmv.x.w` to move the scalar's IEEE-754 bit pattern into a
    /// temporary integer register, then `FBCX.PS f{dest}, tmp`
    /// (`MATCH_FBCX_PS = 0x0000_300b`; I-type, opcode=0x0b, funct3=3, imm=0).
    ///
    /// After this call, `f{dest}` holds `[scalar; 8]` in PS interpretation,
    /// ready to be used as the `scratch` argument of [`fmul_ps_row`] or
    /// [`scale_c_row`].
    ///
    /// # Safety
    /// Requires the `f` target feature (guaranteed by the module gate).
    /// `dest` must not hold live C-tile data that must be preserved; if it
    /// does, spill and restore it around this call (see module doc).
    #[inline(always)]
    pub unsafe fn broadcast_ps(scalar: f32, dest: u8) {
        let tmp: u64;
        asm!(
            "fmv.x.w {tmp}, {x}",
            x   = in(freg) scalar,
            tmp = out(reg) tmp,
            options(nostack, preserves_flags),
        );
        // FBCX.PS f{dest}, {tmp}: broadcasts bit pattern of tmp into all 8 PS lanes.
        match dest {
            0  => asm!(".insn i 0x0b, 3, f0,  {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            1  => asm!(".insn i 0x0b, 3, f1,  {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            2  => asm!(".insn i 0x0b, 3, f2,  {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            3  => asm!(".insn i 0x0b, 3, f3,  {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            4  => asm!(".insn i 0x0b, 3, f4,  {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            5  => asm!(".insn i 0x0b, 3, f5,  {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            6  => asm!(".insn i 0x0b, 3, f6,  {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            7  => asm!(".insn i 0x0b, 3, f7,  {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            8  => asm!(".insn i 0x0b, 3, f8,  {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            9  => asm!(".insn i 0x0b, 3, f9,  {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            10 => asm!(".insn i 0x0b, 3, f10, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            11 => asm!(".insn i 0x0b, 3, f11, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            12 => asm!(".insn i 0x0b, 3, f12, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            13 => asm!(".insn i 0x0b, 3, f13, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            14 => asm!(".insn i 0x0b, 3, f14, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            15 => asm!(".insn i 0x0b, 3, f15, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            16 => asm!(".insn i 0x0b, 3, f16, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            17 => asm!(".insn i 0x0b, 3, f17, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            18 => asm!(".insn i 0x0b, 3, f18, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            19 => asm!(".insn i 0x0b, 3, f19, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            20 => asm!(".insn i 0x0b, 3, f20, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            21 => asm!(".insn i 0x0b, 3, f21, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            22 => asm!(".insn i 0x0b, 3, f22, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            23 => asm!(".insn i 0x0b, 3, f23, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            24 => asm!(".insn i 0x0b, 3, f24, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            25 => asm!(".insn i 0x0b, 3, f25, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            26 => asm!(".insn i 0x0b, 3, f26, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            27 => asm!(".insn i 0x0b, 3, f27, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            28 => asm!(".insn i 0x0b, 3, f28, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            29 => asm!(".insn i 0x0b, 3, f29, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            30 => asm!(".insn i 0x0b, 3, f30, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            31 => asm!(".insn i 0x0b, 3, f31, {t}, 0", t = in(reg) tmp, options(nostack, preserves_flags)),
            _  => {}
        }
    }

    /// Scale the PS register pair for `row` by the pre-broadcast PS register `scratch`.
    ///
    /// Issues two `FMUL.PS` instructions (`MATCH_FMUL_PS = 0x1000_007b`;
    /// R-type, opcode=0x7b, funct7=0x08, funct3=0):
    ///
    /// ```text
    /// f[2*row]   = f[2*row]   .* f[scratch]
    /// f[2*row+1] = f[2*row+1] .* f[scratch]
    /// ```
    ///
    /// The caller must ensure `f[scratch]` holds `[alpha; 8]` before calling
    /// this function -- typically via a preceding [`broadcast_ps`]`(alpha, scratch)`.
    /// When scaling multiple rows with the same `alpha`, call [`broadcast_ps`]
    /// once, then call [`fmul_ps_row`] for each row.
    ///
    /// # Panics (debug builds)
    /// Fires a `debug_assert` if `scratch` equals `2*row` or `2*row+1`
    /// (the scratch register would overwrite the row data being scaled).
    ///
    /// # Safety
    /// Call `tensor_wait(TensorEvent::Fma)` before this function; the tensor
    /// co-processor must have finished writing the FP register file before
    /// any PS operations read it.
    #[inline(always)]
    pub unsafe fn fmul_ps_row(row: u32, scratch: u8) {
        debug_assert!(
            scratch != 2 * row as u8 && scratch != 2 * row as u8 + 1,
            "fmul_ps_row: scratch f{} conflicts with C-tile row {} (f{} and f{})",
            scratch,
            row,
            2 * row,
            2 * row + 1,
        );
        match row {
            0  => scale_row!((0,  1),  scratch),
            1  => scale_row!((2,  3),  scratch),
            2  => scale_row!((4,  5),  scratch),
            3  => scale_row!((6,  7),  scratch),
            4  => scale_row!((8,  9),  scratch),
            5  => scale_row!((10, 11), scratch),
            6  => scale_row!((12, 13), scratch),
            7  => scale_row!((14, 15), scratch),
            8  => scale_row!((16, 17), scratch),
            9  => scale_row!((18, 19), scratch),
            10 => scale_row!((20, 21), scratch),
            11 => scale_row!((22, 23), scratch),
            12 => scale_row!((24, 25), scratch),
            13 => scale_row!((26, 27), scratch),
            14 => scale_row!((28, 29), scratch),
            15 => scale_row!((30, 31), scratch),
            _  => {}
        }
    }

    /// Broadcast `alpha` into `f[scratch]`, then scale the PS register pair for `row`.
    ///
    /// Convenience wrapper: calls [`broadcast_ps`]`(alpha, scratch)` then
    /// [`fmul_ps_row`]`(row, scratch)`. The broadcast is repeated on every call;
    /// when scaling multiple rows with the same `alpha`, call [`broadcast_ps`]
    /// once and [`fmul_ps_row`] for each row instead.
    ///
    /// Pass [`PS_SCRATCH_DEFAULT`] (28) for `scratch` when the C tile has at
    /// most 14 rows. For full 16-row tiles, see the module-level doc for the
    /// spill/restore pattern.
    ///
    /// # Safety
    /// Call `tensor_wait(TensorEvent::Fma)` before this function; the tensor
    /// co-processor must have finished writing the FP register file.
    #[inline(always)]
    pub unsafe fn scale_c_row(row: u32, alpha: f32, scratch: u8) {
        broadcast_ps(alpha, scratch);
        fmul_ps_row(row, scratch);
    }
}

// Re-export the inner items when the feature is present.
#[cfg(target_arch = "riscv64")]
#[cfg(target_feature = "f")]
pub use inner::*;
