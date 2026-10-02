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
//! or lanes 0 and 1) and are therefore insufficient; the full-width
//! [`store_ps`](crate::simd::store_ps)/[`load_ps`](crate::simd::load_ps) pair
//! (FSQ2/FLQ2) preserves all 256 bits and should be used instead.
//!
//! # Transcendental functions
//!
//! [`fexp_ps`](crate::simd::fexp_ps) (2^x), [`flog_ps`](crate::simd::flog_ps)
//! (log2 x) and [`frcp_ps`](crate::simd::frcp_ps) (1/x) execute natively,
//! within 1 ULP with round-towards-zero (PRM). `FSIN.PS`, `FRSQ.PS`,
//! `FDIV.PS` and `FSQRT.PS` are not wrapped: they trap to M-mode emulation
//! (mcause 30) and are unsuitable for inner loops.
//!
//! # Lane masking
//!
//! PS arithmetic and `FLW.PS`/`FSW.PS` update only the lanes enabled in mask
//! register m0; FLQ2/FSQ2 are unmasked. At kernel entry m0 enables all eight
//! lanes (confirmed by `simd-test-rs`).
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
//! Source: `gdb/include/opcode/esperanto-opc.h` in the ET-SoC-1 binutils fork;
//! the transcendental and FLQ2/FSQ2 rows follow the PRM field tables.
//! All PS arithmetic instructions use the RISC-V custom-3 opcode space (0x7b).
//!
//! | Instruction  | MATCH          | Format | Notes |
//! |---|---|---|---|
//! | `fmul.ps`    | `0x1000_007b`  | R-type | fd = fs1 .* fs2 (element-wise). funct7=0x08, funct3=0. |
//! | `fbc.ps`     | `0x0000_000b`  | I-type | Load 4 B from `rs1+imm`, broadcast to all 8 lanes of fd. opcode=0x0b, funct3=0. |
//! | `fbcx.ps`    | `0x0000_300b`  | I-type | Broadcast GPR `rs1` to all 8 lanes of fd; imm=0 fixed. opcode=0x0b, funct3=3. |
//! | `fmvs.x.ps`  | `0xe000_207b`  | R-type | Extract lane `rs2[2:0]` from PS register `rs1` to GPR `rd`. funct7=0x70, funct3=2. |
//! | `fexp.ps`    | `0x5840_007b`  | R-type | fd = 2^fs1. funct7=0x2c, rs2=4, funct3=0. |
//! | `flog.ps`    | `0x5830_007b`  | R-type | fd = log2(fs1). funct7=0x2c, rs2=3, funct3=0. |
//! | `frcp.ps`    | `0x5870_007b`  | R-type | fd = 1/fs1. funct7=0x2c, rs2=7, funct3=0. |
//! | `flq2`       | `0x0000_5007`  | I-type | Load 32 B from `rs1+imm` into all 8 lanes of fd; unmasked. opcode=0x07, funct3=5. |
//! | `fsq2`       | `0x0000_5027`  | S-type | Store all 8 lanes of fs2 to `rs1+imm`; unmasked. opcode=0x27, funct3=5. |

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

    // Dispatch a runtime FP register number (0..=31) to `$emit!(N, args...)`
    // with N as an integer literal, so that `$emit` can splice it into the
    // assembly template with `concat!`. With a constant register argument and
    // `#[inline(always)]`, the match folds to the single selected arm.
    #[rustfmt::skip] // 32-way register dispatch; keep one compact arm per register
    macro_rules! dispatch_register {
        ($register:expr, $emit:ident $(, $arg:expr)*) => {
            match $register {
                0  => $emit!(0  $(, $arg)*),
                1  => $emit!(1  $(, $arg)*),
                2  => $emit!(2  $(, $arg)*),
                3  => $emit!(3  $(, $arg)*),
                4  => $emit!(4  $(, $arg)*),
                5  => $emit!(5  $(, $arg)*),
                6  => $emit!(6  $(, $arg)*),
                7  => $emit!(7  $(, $arg)*),
                8  => $emit!(8  $(, $arg)*),
                9  => $emit!(9  $(, $arg)*),
                10 => $emit!(10 $(, $arg)*),
                11 => $emit!(11 $(, $arg)*),
                12 => $emit!(12 $(, $arg)*),
                13 => $emit!(13 $(, $arg)*),
                14 => $emit!(14 $(, $arg)*),
                15 => $emit!(15 $(, $arg)*),
                16 => $emit!(16 $(, $arg)*),
                17 => $emit!(17 $(, $arg)*),
                18 => $emit!(18 $(, $arg)*),
                19 => $emit!(19 $(, $arg)*),
                20 => $emit!(20 $(, $arg)*),
                21 => $emit!(21 $(, $arg)*),
                22 => $emit!(22 $(, $arg)*),
                23 => $emit!(23 $(, $arg)*),
                24 => $emit!(24 $(, $arg)*),
                25 => $emit!(25 $(, $arg)*),
                26 => $emit!(26 $(, $arg)*),
                27 => $emit!(27 $(, $arg)*),
                28 => $emit!(28 $(, $arg)*),
                29 => $emit!(29 $(, $arg)*),
                30 => $emit!(30 $(, $arg)*),
                31 => $emit!(31 $(, $arg)*),
                _  => unreachable!(),
            }
        };
    }

    // FLQ2 f{n}, 0(addr): I-type, opcode 0x07 (LOAD-FP), funct3 5.
    macro_rules! emit_load_ps {
        ($n:literal, $addr:expr) => {
            fp_asm!(
                concat!(".insn i 0x07, 5, f", $n, ", 0({a})"),
                a = in(reg) $addr,
                options(nostack, preserves_flags),
            )
        };
    }

    // FSQ2 f{n}, 0(addr): S-type, opcode 0x27 (STORE-FP), funct3 5.
    macro_rules! emit_store_ps {
        ($n:literal, $addr:expr) => {
            fp_asm!(
                concat!(".insn s 0x27, 5, f", $n, ", 0({a})"),
                a = in(reg) $addr,
                options(nostack, preserves_flags),
            )
        };
    }

    // Unary PS transcendental f{n} = op(f{n}): R-type, opcode 0x7b, funct3 0,
    // funct7 0x2c (funct5 01011, fmt 00); the rs2 field selects the function
    // and is written as the FP register of that number.
    macro_rules! emit_unary_ps {
        ($n:literal, $selector:literal) => {
            fp_asm!(
                concat!(".insn r 0x7b, 0, 44, f", $n, ", f", $n, ", f", $selector),
                options(nomem, nostack, preserves_flags),
            )
        };
    }
    macro_rules! emit_fexp_ps {
        ($n:literal) => {
            emit_unary_ps!($n, 4)
        };
    }
    macro_rules! emit_flog_ps {
        ($n:literal) => {
            emit_unary_ps!($n, 3)
        };
    }
    macro_rules! emit_frcp_ps {
        ($n:literal) => {
            emit_unary_ps!($n, 7)
        };
    }

    /// Loads eight consecutive f32 values from `addr` into PS register
    /// `register` (0..=31), lane `i` from `addr + 4 * i`, with `FLQ2`
    /// (I-type, opcode 0x07, funct3 5).
    ///
    /// FLQ2 is a full-width 256-bit load and is not subject to the mask
    /// register m0: all eight lanes are written.
    ///
    /// # Panics
    /// If `register > 31`.
    ///
    /// # Safety
    /// `addr` must be 32-byte aligned and `[addr, addr + 32)` readable device
    /// memory. `register` must not hold live data that must be preserved.
    #[inline(always)]
    pub unsafe fn load_ps(register: u8, addr: usize) {
        assert!(register < 32, "load_ps: f{register} is not an FP register");
        // SAFETY: the caller guarantees a readable, aligned source and a free
        // destination register; `fp_asm!` declares every FP register clobbered.
        unsafe { dispatch_register!(register, emit_load_ps, addr) }
    }

    /// Stores the eight f32 lanes of PS register `register` (0..=31) to
    /// `addr`, lane `i` to `addr + 4 * i`, with `FSQ2` (S-type, opcode 0x27,
    /// funct3 5).
    ///
    /// FSQ2 is a full-width 256-bit store and is not subject to m0. It
    /// preserves all 256 bits, so a `store_ps`/[`load_ps`] pair can spill and
    /// restore a scratch register over a full 16-row C tile. The store is an
    /// ordinary cached store: data destined for the host or another shire must
    /// afterwards be written back with `cache::cache_writeback`.
    ///
    /// # Panics
    /// If `register > 31`.
    ///
    /// # Safety
    /// `addr` must be 32-byte aligned and `[addr, addr + 32)` writable device
    /// memory not concurrently accessed by another hart.
    #[inline(always)]
    pub unsafe fn store_ps(register: u8, addr: usize) {
        assert!(register < 32, "store_ps: f{register} is not an FP register");
        // SAFETY: the caller guarantees a writable, aligned, exclusive
        // destination; the instruction reads only f{register}.
        unsafe { dispatch_register!(register, emit_store_ps, addr) }
    }

    /// Replaces each lane `x` of PS register `register` (0..=31) by `2^x`,
    /// with `FEXP.PS` (funct7 0x2c, rs2 4).
    ///
    /// Native, within 1 ULP, round towards zero (PRM). Subnormal inputs are
    /// treated as zero and subnormal results flushed to zero. Inputs below
    /// -126.0, and -inf, give +0; inputs of at least 128.0, and +inf, give
    /// +inf; +/-0 gives 1. A signalling NaN gives the default NaN and raises
    /// the invalid flag. For the natural exponential, scale the argument by
    /// log2(e) beforehand.
    ///
    /// Lanes inactive in mask register m0 are left unchanged.
    ///
    /// # Panics
    /// If `register > 31`.
    ///
    /// # Safety
    /// `register` must hold the operand; it is overwritten. Call
    /// `tensor_wait(TensorEvent::Fma)` first if the tensor co-processor wrote it.
    #[inline(always)]
    pub unsafe fn fexp_ps(register: u8) {
        assert!(register < 32, "fexp_ps: f{register} is not an FP register");
        // SAFETY: writes only f{register}, which the caller has designated.
        unsafe { dispatch_register!(register, emit_fexp_ps) }
    }

    /// Replaces each lane `x` of PS register `register` (0..=31) by
    /// `log2(x)`, with `FLOG.PS` (funct7 0x2c, rs2 3).
    ///
    /// Native, within 1 ULP, round towards zero (PRM). Subnormal inputs are
    /// treated as zero. Negative inputs, including -inf, give NaN and raise
    /// the invalid flag; +/-0 gives -inf; +1 gives +0; +inf gives +inf.
    ///
    /// Lanes inactive in m0 are left unchanged.
    ///
    /// # Panics
    /// If `register > 31`.
    ///
    /// # Safety
    /// As for [`fexp_ps`].
    #[inline(always)]
    pub unsafe fn flog_ps(register: u8) {
        assert!(register < 32, "flog_ps: f{register} is not an FP register");
        // SAFETY: writes only f{register}, which the caller has designated.
        unsafe { dispatch_register!(register, emit_flog_ps) }
    }

    /// Replaces each lane `x` of PS register `register` (0..=31) by `1/x`,
    /// with `FRCP.PS` (funct7 0x2c, rs2 7).
    ///
    /// Native, within 1 ULP, round towards zero (PRM). Subnormal inputs are
    /// treated as zero and subnormal results flushed to zero. +/-0 gives
    /// +/-inf; +/-inf gives +/-0; inputs of magnitude above 2^126 give +/-0.
    /// Unlike `FDIV.PS`, which traps to M-mode emulation, FRCP.PS executes in
    /// hardware and is the preferred reciprocal in inner loops.
    ///
    /// Lanes inactive in m0 are left unchanged.
    ///
    /// # Panics
    /// If `register > 31`.
    ///
    /// # Safety
    /// As for [`fexp_ps`].
    #[inline(always)]
    pub unsafe fn frcp_ps(register: u8) {
        assert!(register < 32, "frcp_ps: f{register} is not an FP register");
        // SAFETY: writes only f{register}, which the caller has designated.
        unsafe { dispatch_register!(register, emit_frcp_ps) }
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
