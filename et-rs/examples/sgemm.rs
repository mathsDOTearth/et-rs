//! Host-side sGEMM demonstration.
//!
//! Computes C = A * B on the ET-SoC-1 tensor extension, then downloads C and
//! verifies a sample of results against a scalar reference computed on the host.
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
const N: usize = 64;  // multiple of GEMM_TILE_N = 16
const K: usize = 64;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let elf_path  = args.next().expect("usage: sgemm <sgemm-rs.elf> [n_shires]");
    let n_shires  = args.next().map_or(1u32, |s| s.parse().expect("n_shires must be u32"));

    let elf    = std::fs::read(&elf_path).map_err(|e| Error::io("read sgemm ELF", e))?;
    let dev    = Device::open(0)?;
    let kernel = dev.load_kernel(&elf)?;

    // Leading dimensions in bytes (row-stride), padded to 64 bytes.
    let lda = ((K * 4).next_multiple_of(64)) as u32;
    let ldb = ((N * 4).next_multiple_of(64)) as u32;
    let ldc = ((N * 4).next_multiple_of(64)) as u32;

    // Allocate matrices using the tensor-aligned allocator.
    let (a_addr, _) = blas::alloc_tensor_matrix(&dev, M, K)?;
    let (b_addr, _) = blas::alloc_tensor_matrix(&dev, K, N)?;
    let (c_addr, _) = blas::alloc_tensor_matrix(&dev, M, N)?;

    // Initialise A and B with simple data on the host.
    let a_host: Vec<f32> = (0..M * K).map(|i| (i % 16) as f32 * 0.1).collect();
    let b_host: Vec<f32> = (0..K * N).map(|i| (i % 8)  as f32 * 0.5).collect();

    // Upload A and B to device memory (row-major, no stride padding in host data).
    upload_matrix(&dev, a_addr, &a_host, M, K, lda)?;
    upload_matrix(&dev, b_addr, &b_host, K, N, ldb)?;

    // Launch the sGEMM kernel.
    blas::sgemm(
        &dev, &kernel,
        M as u32, N as u32, K as u32,
        1.0, a_addr, lda,
             b_addr, ldb,
        0.0, c_addr, ldc,
        n_shires,
    )?;
    println!("sGEMM kernel returned.");

    // Download C and verify a corner element.
    let c_flat = download_matrix(&dev, c_addr, M, N, ldc)?;

    // Reference: C[0][0] = sum_k A[0][k] * B[k][0]
    let ref_00: f32 = (0..K).map(|kk| a_host[kk] * b_host[kk * N]).sum();
    let dev_00 = c_flat[0];
    let err_00 = (ref_00 - dev_00).abs();
    println!("C[0][0]: reference = {ref_00:.4}, device = {dev_00:.4}, |err| = {err_00:.2e}");
    assert!(
        err_00 < 1e-3 * ref_00.abs().max(1.0),
        "C[0][0] exceeds tolerance: ref={ref_00}, dev={dev_00}"
    );

    // Spot-check a few more elements.
    let i = 3;
    let j = 7;
    let ref_ij: f32 = (0..K).map(|kk| a_host[i * K + kk] * b_host[kk * N + j]).sum();
    let dev_ij = c_flat[i * N + j];
    let err_ij = (ref_ij - dev_ij).abs();
    println!("C[{i}][{j}]: reference = {ref_ij:.4}, device = {dev_ij:.4}, |err| = {err_ij:.2e}");
    assert!(
        err_ij < 1e-3 * ref_ij.abs().max(1.0),
        "C[{i}][{j}] exceeds tolerance"
    );

    println!("All spot-checks passed.");
    Ok(())
}

/// Upload a row-major matrix to device memory, inserting stride padding.
fn upload_matrix(
    dev:  &Device<et_soc1::transport::IoctlTransport>,
    addr: u64,
    data: &[f32],
    rows: usize,
    cols: usize,
    lda:  u32,
) -> Result<()> {
    let row_bytes = lda as u64;
    for r in 0..rows {
        let src_start = r * cols;
        let src_row   = &data[src_start..src_start + cols];
        // SAFETY: f32 is DevicePod.
        let src_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(src_row.as_ptr() as *const u8, cols * 4)
        };
        dev.memcpy_h2d(src_bytes, addr + r as u64 * row_bytes)?;
    }
    Ok(())
}

/// Download a strided matrix from device memory into a contiguous host Vec.
fn download_matrix(
    dev:  &Device<et_soc1::transport::IoctlTransport>,
    addr: u64,
    rows: usize,
    cols: usize,
    ldc:  u32,
) -> Result<Vec<f32>> {
    let mut out     = vec![0.0f32; rows * cols];
    let row_bytes = ldc as u64;
    let mut raw   = vec![0u8; ldc as usize];

    for r in 0..rows {
        dev.memcpy_d2h(addr + r as u64 * row_bytes, &mut raw)?;
        // SAFETY: raw contains f32 values in device little-endian byte order,
        // matching the host (both are little-endian).
        for c in 0..cols {
            out[r * cols + c] = f32::from_le_bytes([
                raw[c * 4],
                raw[c * 4 + 1],
                raw[c * 4 + 2],
                raw[c * 4 + 3],
            ]);
        }
    }
    Ok(out)
}
