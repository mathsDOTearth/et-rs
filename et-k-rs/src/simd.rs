//! Packed-single (PS) SIMD intrinsics for the ET-SoC-1 Minion FP register file.
//!
//! The ET-SoC-1 PS extension operates on the 256-bit FP registers (f0..f31),
//! treating each as a vector of eight single-precision (f32) lanes. PS
//! instructions use the standard RISC-V floating-point register file and
//! therefore require the `f` target feature to be enabled at compile time.
//!
//! # Enabling this module
//!
//! The entire module is gated on `cfg(target_feature = "f")`. To compile it,
//! pass `+f` via `RUSTFLAGS` or `.cargo/config.toml`:
//!
//! ```toml
//! [target.riscv64imac-unknown-none-elf]
//! rustflags = ["-C", "target-feature=+f"]
//! ```
//!
//! Without `+f` the module is empty; references from other modules must be
//! similarly gated or the kernel must be compiled with `+f` enabled globally.
//!
//! # Status
//!
//! The functions below are stubs pending confirmation of the exact PS
//! instruction encodings from the ET-SoC-1 Programmers Reference Manual
//! Chapter 5. Each function calls `unimplemented!()` at runtime to prevent
//! silent wrong results; replace the bodies with the correct PS opcodes once
//! the encodings are verified against the PRM.

#[cfg(target_arch = "riscv64")]
#[cfg(target_feature = "f")]
mod inner {
    /// Multiply every element of FP register pair `(f[2*row], f[2*row+1])`
    /// by `alpha` using PS SIMD, scaling one 64-byte row of C in place.
    ///
    /// `row` must satisfy `row < 16` (the pair is `f[2*row]`, `f[2*row+1]`).
    ///
    /// # Safety
    /// The FP register file must not be written by a TensorFMA32 in flight;
    /// call `tensor_wait(TensorEvent::Fma)` first.
    ///
    /// # TODO
    /// Replace body with `FMUL.PS f[2*row], f[2*row], fs; FMUL.PS
    /// f[2*row+1], f[2*row+1], fs` once the PS opcode encoding is confirmed.
    #[inline(always)]
    pub unsafe fn scale_c_row(_row: u32, _alpha: f32) {
        unimplemented!("scale_c_row: FMUL.PS opcode not yet confirmed from PRM Chapter 5");
    }

    /// Broadcast a scalar `f32` into all eight lanes of a 256-bit FP register.
    ///
    /// Returns the scalar unchanged as a placeholder; the actual implementation
    /// requires `FMVS.PS fd, rs` from the PS extension.
    ///
    /// # Safety
    /// Caller must hold the `f` target feature (guaranteed by module gate).
    ///
    /// # TODO
    /// Replace with `FMVS.PS` once the encoding is confirmed.
    #[inline(always)]
    pub unsafe fn broadcast_ps(_scalar: f32) -> f32 {
        unimplemented!("broadcast_ps: FMVS.PS opcode not yet confirmed from PRM Chapter 5");
    }
}

// Re-export the inner items when the feature is present.
#[cfg(target_arch = "riscv64")]
#[cfg(target_feature = "f")]
pub use inner::*;
