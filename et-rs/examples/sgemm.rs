//! Host-side sGEMM demonstration.
//!
//! Computes C = A * B on the ET-SoC-1 tensor extension, then downloads C and
//! verifies every element against a scalar reference computed on the host.
//!
//! The inputs are pseudo-random rather than periodic, so an error in tile
//! addressing (a wrong row, column or K-block offset) produces a mismatch
//! instead of reading an identical value from the wrong place. C is prefilled
//! with NaN so that an unwritten element also fails.
//!
//! Run against real hardware:
//!   cargo run --example sgemm -- <elf-path> [n_shires]
//!
//! where `<elf-path>` is the compiled `sgemm-rs.elf` device image.

use std::env;

use et_soc1::{Device, Error, Result, blas};

// Tile the problem so all dimensions satisfy the v0.1 constraints:
//   M, K can be arbitrary >= 1; N must be a multiple of 16.
const M: usize = 64;
const N: usize = 64; // multiple of GEMM_TILE_N = 16
const K: usize = 64;

/// Relative tolerance, scaled by `sum_k |A[i][k] * B[k][j]|`. The tensor unit
/// accumulates in f32 in an order different from the host loop; the worst-case
/// reordering error is bounded by about K * 2^-24 (roughly 4e-6 for K = 64),
/// so a genuine addressing error exceeds this bound by orders of magnitude.
const RELATIVE_TOLERANCE: f32 = 1e-4;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let elf_path = args.next().expect("usage: sgemm <sgemm-rs.elf> [n_shires]");
    let n_shires = args
        .next()
        .map_or(1u32, |s| s.parse().expect("n_shires must be u32"));

    let elf = std::fs::read(&elf_path).map_err(|e| Error::io("read sgemm ELF", e))?;
    let dev = Device::open(0)?;
    let kernel = dev.load_kernel(&elf)?;

    // Leading dimensions in bytes (row-stride), padded to 64 bytes.
    let lda = ((K * 4).next_multiple_of(64)) as u32;
    let ldb = ((N * 4).next_multiple_of(64)) as u32;
    let ldc = ((N * 4).next_multiple_of(64)) as u32;

    // Allocate matrices using the tensor-aligned allocator.
    let (a_addr, _) = blas::alloc_tensor_matrix(&dev, M, K)?;
    let (b_addr, _) = blas::alloc_tensor_matrix(&dev, K, N)?;
    let (c_addr, _) = blas::alloc_tensor_matrix(&dev, M, N)?;

    // Pseudo-random inputs in [-1, 1) from a fixed-seed generator.
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let a_host: Vec<f32> = (0..M * K).map(|_| next_uniform(&mut state)).collect();
    let b_host: Vec<f32> = (0..K * N).map(|_| next_uniform(&mut state)).collect();

    // Upload A and B to device memory (row-major, no stride padding in host
    // data) and poison C.
    upload_matrix(&dev, a_addr, &a_host, M, K, lda)?;
    upload_matrix(&dev, b_addr, &b_host, K, N, ldb)?;
    upload_matrix(&dev, c_addr, &vec![f32::NAN; M * N], M, N, ldc)?;

    // Launch the sGEMM kernel.
    blas::sgemm(
        &dev, &kernel, M as u32, N as u32, K as u32, 1.0, a_addr, lda, b_addr, ldb, 0.0, c_addr,
        ldc, n_shires,
    )?;
    println!("sGEMM kernel returned.");

    let c_flat = download_matrix(&dev, c_addr, M, N, ldc)?;

    let mut failures = 0_usize;
    let mut max_relative_error = 0.0_f32;
    for i in 0..M {
        for j in 0..N {
            let (reference, magnitude) = (0..K).fold((0.0_f32, 0.0_f32), |(sum, mag), kk| {
                let product = a_host[i * K + kk] * b_host[kk * N + j];
                (sum + product, mag + product.abs())
            });
            let device = c_flat[i * N + j];
            let relative_error = (reference - device).abs() / magnitude.max(f32::MIN_POSITIVE);
            // A NaN error (unwritten element) fails the check below.
            if relative_error.is_nan() || relative_error > RELATIVE_TOLERANCE {
                if failures < 16 {
                    eprintln!("FAIL C[{i}][{j}]: reference = {reference}, device = {device}");
                }
                failures += 1;
            } else {
                max_relative_error = max_relative_error.max(relative_error);
            }
        }
    }

    if failures == 0 {
        println!(
            "All {} elements within tolerance (max relative error {max_relative_error:.2e}).",
            M * N
        );
        Ok(())
    } else {
        Err(Error::Protocol(format!(
            "sgemm: {failures}/{} elements outside tolerance",
            M * N
        )))
    }
}

/// Returns the next value in [-1, 1) from a 64-bit linear congruential
/// generator (Knuth MMIX constants), using the top 24 bits so that every
/// output is exactly representable as f32.
fn next_uniform(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 40) as f32 / (1u32 << 23) as f32) - 1.0
}

/// Upload a row-major matrix to device memory, inserting stride padding.
fn upload_matrix(
    dev: &Device<et_soc1::transport::IoctlTransport>,
    addr: u64,
    data: &[f32],
    rows: usize,
    cols: usize,
    lda: u32,
) -> Result<()> {
    let row_bytes = lda as u64;
    for r in 0..rows {
        let src_start = r * cols;
        let src_row = &data[src_start..src_start + cols];
        // SAFETY: f32 is DevicePod.
        let src_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(src_row.as_ptr() as *const u8, cols * 4) };
        dev.memcpy_h2d(src_bytes, addr + r as u64 * row_bytes)?;
    }
    Ok(())
}

/// Download a strided matrix from device memory into a contiguous host Vec.
fn download_matrix(
    dev: &Device<et_soc1::transport::IoctlTransport>,
    addr: u64,
    rows: usize,
    cols: usize,
    ldc: u32,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; rows * cols];
    let row_bytes = ldc as u64;
    let mut raw = vec![0u8; ldc as usize];

    for r in 0..rows {
        dev.memcpy_d2h(addr + r as u64 * row_bytes, &mut raw)?;
        // SAFETY: raw contains f32 values in device little-endian byte order,
        // matching the host (both are little-endian).
        for c in 0..cols {
            out[r * cols + c] =
                f32::from_le_bytes([raw[c * 4], raw[c * 4 + 1], raw[c * 4 + 2], raw[c * 4 + 3]]);
        }
    }
    Ok(out)
}
