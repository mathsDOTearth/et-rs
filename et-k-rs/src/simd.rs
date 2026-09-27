//! Packed-single (PS) SIMD intrinsics for the ET-SoC-1 Minion FP register file.
//!
//! The ET-SoC-1 PS extension operates on the 256-bit FP registers (f0..f31),
//! treating each as a vector of eight single-precision (f32) lanes. PS
//! instructions share the standard RISC-V FP register file and therefore
//! require a target with the F extension.
//!
//! # Enabling this module
//!
//! The module is gated on `cfg(et_fp_registers)`, which `build.rs` sets when
//! the target triple includes the F extension or `-C target-feature=+f` is
//! given (as it is in `.cargo/config.toml` for `riscv64imac-unknown-none-elf`).
//! `cfg(target_feature = "f")` is not used: stable rustc does not expose the
//! unstable RISC-V `f` feature to `cfg`, even where it is enabled.
//!
//! Without the F extension the module is empty; callers must be similarly
//! gated.
//!
//! # Register assignment for a 16-row C tile
//!
//! A 16-row GEMM C tile occupies the full FP register file (f0..f31) in
//! register pairs: row N maps to (`f[2N]`, `f[2N+1]`). The broadcast scratch
//! register for [`broadcast_ps`](crate::simd::broadcast_ps) and [`fmul_ps_row`](crate::simd::fmul_ps_row) must not overlap with
//! the pair being scaled.
//!
//! For tiles of at most 14 rows, pass [`PS_SCRATCH_DEFAULT`](crate::simd::PS_SCRATCH_DEFAULT) (f28) as
//! `scratch`. A full 16-row tile leaves no free register: every one of
//! f0..f31 holds C data. A scratch register must then be spilled and
//! restored around the broadcast, and the spill must preserve all 256 bits.
//! The scalar `fsw`/`fsd` instructions save only the low 32 or 64 bits (lane 0,
//! or lanes 0 and 1) and are therefore insufficient; a full-width PS
//! store/load pair is required, which this module does not yet wrap. Until it
//! does, scale tiles of at most 14 rows, or store rows 14 and 15 with
//! `tensor_store` and scale them separately.
//!
//! # Compiler interaction
//!
//! Each instruction here declares every FP register clobbered, so the compiler
//! keeps none of its own values in f0..f31 across a call. The C-tile contents
//! are invisible to the compiler; code between the FMA that produced the tile
//! and the final store must not itself use floating point, or the compiler may
//! allocate a register that holds tile data.
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
#[cfg(et_fp_registers)]
mod inner {
    /// Default PS broadcast scratch register (f28 / ft8).
    ///
    /// Safe for C tiles with at most 14 rows (rows 0..=13). Rows 14 and 15
    /// place C-tile data in f28..f31, conflicting with this value; see the
    /// module documentation for full 16-row tiles.
    pub const PS_SCRATCH_DEFAULT: u8 = 28;

    // Emit two FMUL.PS instructions for a register pair with literal operands.
    // R-type: opcode=0x7b, funct3=0, funct7=8.
    // fd = fs1 = f{lo} or f{hi}; fs2 = f{s}.
    #[rustfmt::skip] // hand-laid .insn operands; keep on one line per instruction
    macro_rules! fmul2 {
        ($lo:literal, $hi:literal, $s:literal) => {
            fp_asm!(
                concat!(
                    ".insn r 0x7b, 0, 8, f", $lo, ", f", $lo, ", f", $s, "\n",
                    ".insn r 0x7b, 0, 8, f", $hi, ", f", $hi, ", f", $s
                ),
                options(nostack, preserves_flags),
            )
        };
    }

    // Dispatch FMUL.PS for register pair ($lo, $hi) over all 32 scratch registers.
    #[rustfmt::skip] // 32-way scratch dispatch; keep one compact arm per register
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
                _  => unreachable!(),
            }
        };
    }

    /// Broadcast the 32-bit pattern `bits` to all eight lanes of PS register
    /// `dest` (0..=31) with `FBCX.PS f{dest}, rs1` (`MATCH_FBCX_PS =
    /// 0x0000_300b`; I-type, opcode=0x0b, funct3=3, imm=0).
    ///
    /// The value travels in an integer register, so no FP register other than
    /// `dest` is written. For an f32 `alpha`, pass `alpha.to_bits()`.
    ///
    /// # Panics
    /// If `dest > 31`.
    ///
    /// # Safety
    /// `dest` must not hold live C-tile data that must be preserved (see the
    /// module documentation).
    #[inline(always)]
    #[rustfmt::skip] // tabular FBCX.PS dispatch; keep one aligned arm per register
    pub unsafe fn broadcast_ps_bits(bits: u32, dest: u8) {
        assert!(dest < 32, "broadcast_ps_bits: dest f{dest} is not an FP register");
        let bits = bits as u64;
        // SAFETY: FBCX.PS writes only f{dest}, which the caller guarantees
        // holds no live data; `fp_asm!` declares every FP register clobbered.
        unsafe { match dest {
            0  => fp_asm!(".insn i 0x0b, 3, f0,  {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            1  => fp_asm!(".insn i 0x0b, 3, f1,  {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            2  => fp_asm!(".insn i 0x0b, 3, f2,  {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            3  => fp_asm!(".insn i 0x0b, 3, f3,  {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            4  => fp_asm!(".insn i 0x0b, 3, f4,  {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            5  => fp_asm!(".insn i 0x0b, 3, f5,  {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            6  => fp_asm!(".insn i 0x0b, 3, f6,  {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            7  => fp_asm!(".insn i 0x0b, 3, f7,  {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            8  => fp_asm!(".insn i 0x0b, 3, f8,  {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            9  => fp_asm!(".insn i 0x0b, 3, f9,  {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            10 => fp_asm!(".insn i 0x0b, 3, f10, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            11 => fp_asm!(".insn i 0x0b, 3, f11, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            12 => fp_asm!(".insn i 0x0b, 3, f12, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            13 => fp_asm!(".insn i 0x0b, 3, f13, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            14 => fp_asm!(".insn i 0x0b, 3, f14, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            15 => fp_asm!(".insn i 0x0b, 3, f15, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            16 => fp_asm!(".insn i 0x0b, 3, f16, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            17 => fp_asm!(".insn i 0x0b, 3, f17, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            18 => fp_asm!(".insn i 0x0b, 3, f18, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            19 => fp_asm!(".insn i 0x0b, 3, f19, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            20 => fp_asm!(".insn i 0x0b, 3, f20, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            21 => fp_asm!(".insn i 0x0b, 3, f21, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            22 => fp_asm!(".insn i 0x0b, 3, f22, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            23 => fp_asm!(".insn i 0x0b, 3, f23, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            24 => fp_asm!(".insn i 0x0b, 3, f24, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            25 => fp_asm!(".insn i 0x0b, 3, f25, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            26 => fp_asm!(".insn i 0x0b, 3, f26, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            27 => fp_asm!(".insn i 0x0b, 3, f27, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            28 => fp_asm!(".insn i 0x0b, 3, f28, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            29 => fp_asm!(".insn i 0x0b, 3, f29, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            30 => fp_asm!(".insn i 0x0b, 3, f30, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            31 => fp_asm!(".insn i 0x0b, 3, f31, {t}, 0", t = in(reg) bits, options(nostack, preserves_flags),),
            _  => unreachable!(),
        } }
    }

    /// Broadcast `scalar` to all eight lanes of PS register `dest` (0..=31).
    ///
    /// Equivalent to [`broadcast_ps_bits`]`(scalar.to_bits(), dest)`. If
    /// `scalar` is computed at run time with FP arithmetic, the compiler may
    /// hold it in an FP register beforehand, which is only safe if no C-tile
    /// data is live yet; a constant or integer-derived `alpha` never touches
    /// the FP register file.
    ///
    /// After this call, `f{dest}` holds `[scalar; 8]` in PS interpretation,
    /// ready to be used as the `scratch` argument of [`fmul_ps_row`] or
    /// [`scale_c_row`].
    ///
    /// # Panics
    /// If `dest > 31`.
    ///
    /// # Safety
    /// As for [`broadcast_ps_bits`].
    #[inline(always)]
    pub unsafe fn broadcast_ps(scalar: f32, dest: u8) {
        // SAFETY: forwarded from the caller.
        unsafe { broadcast_ps_bits(scalar.to_bits(), dest) };
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
    /// # Panics
    /// If `row > 15` or `scratch > 31`. In debug builds, also if `scratch`
    /// equals `2*row` or `2*row+1` (the scratch register would overwrite the
    /// row data being scaled).
    ///
    /// # Safety
    /// Call `tensor_wait(TensorEvent::Fma)` before this function; the tensor
    /// co-processor must have finished writing the FP register file before
    /// any PS operations read it.
    #[inline(always)]
    #[rustfmt::skip] // tabular row-pair dispatch; keep one aligned arm per row
    pub unsafe fn fmul_ps_row(row: u32, scratch: u8) {
        assert!(row < 16, "fmul_ps_row: row {row} exceeds the 16-row C tile");
        assert!(scratch < 32, "fmul_ps_row: scratch f{scratch} is not an FP register");
        debug_assert!(
            scratch != 2 * row as u8 && scratch != 2 * row as u8 + 1,
            "fmul_ps_row: scratch f{} conflicts with C-tile row {} (f{} and f{})",
            scratch,
            row,
            2 * row,
            2 * row + 1,
        );
        // SAFETY: FMUL.PS writes only the row pair, which the caller guarantees
        // holds completed C data; `fp_asm!` declares every FP register
        // clobbered.
        unsafe { match row {
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
            _  => unreachable!(),
        } }
    }

    /// Broadcast `alpha` into `f[scratch]`, then scale the PS register pair for `row`.
    ///
    /// Convenience wrapper: calls [`broadcast_ps`]`(alpha, scratch)` then
    /// [`fmul_ps_row`]`(row, scratch)`. The broadcast is repeated on every call;
    /// when scaling multiple rows with the same `alpha`, call [`broadcast_ps`]
    /// once and [`fmul_ps_row`] for each row instead.
    ///
    /// Pass [`PS_SCRATCH_DEFAULT`] (28) for `scratch` when the C tile has at
    /// most 14 rows. For full 16-row tiles, see the module documentation.
    ///
    /// # Panics
    /// As for [`broadcast_ps`] and [`fmul_ps_row`].
    ///
    /// # Safety
    /// Call `tensor_wait(TensorEvent::Fma)` before this function; the tensor
    /// co-processor must have finished writing the FP register file.
    #[inline(always)]
    pub unsafe fn scale_c_row(row: u32, alpha: f32, scratch: u8) {
        // SAFETY: forwarded from the caller.
        unsafe {
            broadcast_ps(alpha, scratch);
            fmul_ps_row(row, scratch);
        }
    }
}

// Re-export the inner items when the feature is present.
#[cfg(target_arch = "riscv64")]
#[cfg(et_fp_registers)]
pub use inner::*;
