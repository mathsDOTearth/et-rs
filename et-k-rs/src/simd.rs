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
//! # Encodings
//!
//! Source: `gdb/include/opcode/esperanto-opc.h` in the ET-SoC-1 binutils fork.
//! All PS arithmetic instructions use the RISC-V custom-3 opcode space (0x7b).
//!
//! | Instruction | MATCH | Format | Notes |
//! |---|---|---|---|
//! | `fmul.ps`  | `0x1000_007b` | R-type | fd = fs1 .* fs2 (element-wise). funct7=0x08, funct3=0. |
//! | `fbc.ps`   | `0x0000_000b` | I-type | Load 4 B from `rs1+imm`, broadcast to all 8 lanes of fd. opcode=0x0b, funct3=0. |
//! | `fbcx.ps`  | `0x0000_300b` | I-type | Broadcast GPR `rs1` to all 8 lanes of fd; imm fixed to 0. opcode=0x0b, funct3=3. |
//! | `fmvs.x.ps`| `0xe000_207b` | R-type | Extract lane `rs2[2:0]` from PS register `rs1` to GPR `rd`. funct7=0x70, funct3=2. |

#[cfg(target_arch = "riscv64")]
#[cfg(target_feature = "f")]
mod inner {
    use core::arch::asm;

    /// Broadcast `scalar` to all eight lanes of the scratch PS register `f28`
    /// (ABI name `ft8`), preparing it for [`scale_c_row`].
    ///
    /// Issues `fmv.x.w` to move the scalar's IEEE-754 bit pattern into a
    /// temporary integer register, then `FBCX.PS f28, tmp`
    /// (`MATCH_FBCX_PS = 0x0000_300b`; I-type, opcode=0x0b, funct3=3, imm=0)
    /// to replicate those four bytes into all eight lanes of `f28`.
    ///
    /// Returns `scalar` unchanged; the side effect is that `f28` holds
    /// `[scalar; 8]` in PS interpretation, ready for element-wise multiply.
    ///
    /// # Constraint
    /// `f28` must not hold live C-tile data. This holds when the C tile has
    /// at most 14 rows (sgemm `arows` <= 13 in PRM 0-indexed notation); a
    /// 16-row full tile uses f0..f31 and leaves no spare PS register.
    ///
    /// # Safety
    /// Requires the `f` target feature (guaranteed by the module gate).
    #[inline(always)]
    pub unsafe fn broadcast_ps(scalar: f32) -> f32 {
        let tmp: u64;
        // fmv.x.w: moves the bit pattern of scalar (in FP register) into an
        //          integer register.
        // FBCX.PS f28, {tmp}: I-type, opcode=0x0b, funct3=3, rd=f28, rs1={tmp}, imm=0.
        //   Broadcasts the 32-bit integer value of {tmp} to all 8 lanes of f28.
        asm!(
            "fmv.x.w {tmp}, {x}",
            ".insn i 0x0b, 3, f28, {tmp}, 0",
            x   = in(freg) scalar,
            tmp = out(reg) tmp,
            options(nostack, preserves_flags),
        );
        scalar
    }

    /// Scale one PS register pair of the C tile by the scalar previously
    /// broadcast into `f28` via [`broadcast_ps`].
    ///
    /// Broadcasts `alpha` into `f28` first (so [`broadcast_ps`] need not be
    /// called separately), then issues two `FMUL.PS` instructions
    /// (`MATCH_FMUL_PS = 0x1000_007b`; R-type, opcode=0x7b, funct7=0x08,
    /// funct3=0), one each for `f[2*row]` and `f[2*row+1]`:
    ///
    /// ```text
    /// f[2*row]   = f[2*row]   .* f28
    /// f[2*row+1] = f[2*row+1] .* f28
    /// ```
    ///
    /// `row` must be in `0..=13`. Rows 14 and 15 use `f28..=f31` as C-tile
    /// data and conflict with the `f28` scratch register; a `debug_assert`
    /// fires in debug builds (the function is a no-op in release for
    /// `row >= 14`).
    ///
    /// # Safety
    /// Call `tensor_wait(TensorEvent::Fma)` before this function; the tensor
    /// co-processor must have finished writing the FP register file before
    /// any PS operations read it.
    #[inline(always)]
    pub unsafe fn scale_c_row(row: u32, alpha: f32) {
        debug_assert!(
            row <= 13,
            "scale_c_row: row must be <= 13; rows 14-15 use f28..f31 as C-tile data"
        );
        // Broadcast alpha to all 8 lanes of f28 (scratch PS register).
        // FBCX.PS f28, {tmp}: opcode=0x0b, funct3=3, rd=f28, rs1={tmp}, imm=0.
        let tmp: u64;
        asm!(
            "fmv.x.w {tmp}, {a}",
            ".insn i 0x0b, 3, f28, {tmp}, 0",
            a   = in(freg) alpha,
            tmp = out(reg) tmp,
            options(nostack, preserves_flags),
        );
        // FMUL.PS fd, fs1, fs2: R-type, opcode=0x7b, funct3=0, funct7=8.
        // Encoding: 0x1000_007b | (fs2 << 20) | (fs1 << 15) | (fd << 7).
        // For each row, fd = fs1 = f[2*row] (or f[2*row+1]), fs2 = f28.
        match row {
            0 => asm!(
                ".insn r 0x7b, 0, 8, f0,  f0,  f28",
                ".insn r 0x7b, 0, 8, f1,  f1,  f28",
                options(nostack, preserves_flags)
            ),
            1 => asm!(
                ".insn r 0x7b, 0, 8, f2,  f2,  f28",
                ".insn r 0x7b, 0, 8, f3,  f3,  f28",
                options(nostack, preserves_flags)
            ),
            2 => asm!(
                ".insn r 0x7b, 0, 8, f4,  f4,  f28",
                ".insn r 0x7b, 0, 8, f5,  f5,  f28",
                options(nostack, preserves_flags)
            ),
            3 => asm!(
                ".insn r 0x7b, 0, 8, f6,  f6,  f28",
                ".insn r 0x7b, 0, 8, f7,  f7,  f28",
                options(nostack, preserves_flags)
            ),
            4 => asm!(
                ".insn r 0x7b, 0, 8, f8,  f8,  f28",
                ".insn r 0x7b, 0, 8, f9,  f9,  f28",
                options(nostack, preserves_flags)
            ),
            5 => asm!(
                ".insn r 0x7b, 0, 8, f10, f10, f28",
                ".insn r 0x7b, 0, 8, f11, f11, f28",
                options(nostack, preserves_flags)
            ),
            6 => asm!(
                ".insn r 0x7b, 0, 8, f12, f12, f28",
                ".insn r 0x7b, 0, 8, f13, f13, f28",
                options(nostack, preserves_flags)
            ),
            7 => asm!(
                ".insn r 0x7b, 0, 8, f14, f14, f28",
                ".insn r 0x7b, 0, 8, f15, f15, f28",
                options(nostack, preserves_flags)
            ),
            8 => asm!(
                ".insn r 0x7b, 0, 8, f16, f16, f28",
                ".insn r 0x7b, 0, 8, f17, f17, f28",
                options(nostack, preserves_flags)
            ),
            9 => asm!(
                ".insn r 0x7b, 0, 8, f18, f18, f28",
                ".insn r 0x7b, 0, 8, f19, f19, f28",
                options(nostack, preserves_flags)
            ),
            10 => asm!(
                ".insn r 0x7b, 0, 8, f20, f20, f28",
                ".insn r 0x7b, 0, 8, f21, f21, f28",
                options(nostack, preserves_flags)
            ),
            11 => asm!(
                ".insn r 0x7b, 0, 8, f22, f22, f28",
                ".insn r 0x7b, 0, 8, f23, f23, f28",
                options(nostack, preserves_flags)
            ),
            12 => asm!(
                ".insn r 0x7b, 0, 8, f24, f24, f28",
                ".insn r 0x7b, 0, 8, f25, f25, f28",
                options(nostack, preserves_flags)
            ),
            13 => asm!(
                ".insn r 0x7b, 0, 8, f26, f26, f28",
                ".insn r 0x7b, 0, 8, f27, f27, f28",
                options(nostack, preserves_flags)
            ),
            // Rows 14 (f28, f29) and 15 (f30, f31) conflict with the f28
            // scratch. The debug_assert above fires in debug builds; in
            // release this arm is a no-op.
            _ => {}
        }
    }
}

// Re-export the inner items when the feature is present.
#[cfg(target_arch = "riscv64")]
#[cfg(target_feature = "f")]
pub use inner::*;
