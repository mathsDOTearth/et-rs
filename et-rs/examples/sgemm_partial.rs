//! Partial-N sGEMM verification.
//!
//! Verifies that `sgemm` produces correct results when N is not a multiple of
//! the tile width (16). The kernel computes `ceil(N / 16)` column tiles; the
//! last partial tile uses the row-stride padding already allocated by
//! `alloc_tensor_matrix` and stores 64 bytes per C row regardless. Only
//! C[row][0..N] is read and verified; the padding bytes are ignored.
//!
//! Run against real hardware:
//!   cargo run --example sgemm_partial -- <sgemm-rs.elf> [n_shires]
//!
//! This example uses `A[i][k] = (i+1) as f32` and `B[k][j] = (j+1) as f32`,
//! giving `C[i][j] = K * (i+1) * (j+1)` -- exact in f32 so zero-tolerance
//! checks are valid.

use std::env;

use et_soc1::{Device, Error, Result, blas};

const M: usize = 32;
const N: usize = 20; // not a multiple of 16: tile 0 = cols 0..16, tile 1 = cols 16..20
const K: usize = 32;

fn main() -> Result<()> {
    let mut args = env::args().skip(1);
    let elf_path = args.next()
        .expect("usage: sgemm_partial <sgemm-rs.elf> [n_shires]");
    let n_shires = args.next()
        .map_or(1u32, |s| s.parse().expect("n_shires must be u32"));

    let elf    = std::fs::read(&elf_path).map_err(|e| Error::io("read sgemm ELF", e))?;
    let dev    = Device::open(0)?;
    let kernel = dev.load_kernel(&elf)?;

    // Row strides in bytes, padded to 64 bytes.
    // For N=20: ldb = ldc = ceil(20*4/64)*64 = 128 bytes per row (32 f32 slots,
    // of which only the first 20 hold user data; the rest are allocated padding).
    let lda = ((K * 4).next_multiple_of(64)) as u32;
    let ldb = ((N * 4).next_multiple_of(64)) as u32;
    let ldc = ((N * 4).next_multiple_of(64)) as u32;

    let (a_addr, _) = blas::alloc_tensor_matrix(&dev, M, K)?;
    let (b_addr, _) = blas::alloc_tensor_matrix(&dev, K, N)?;
    let (c_addr, _) = blas::alloc_tensor_matrix(&dev, M, N)?;

    // A[i][k] = (i+1) as f32; B[k][j] = (j+1) as f32.
    // Exact reference: C[i][j] = K * (i+1) * (j+1).
    let a_host: Vec<f32> = (0..M * K).map(|idx| (idx / K + 1) as f32).collect();
    let b_host: Vec<f32> = (0..K * N).map(|idx| (idx % N + 1) as f32).collect();

    upload_matrix(&dev, a_addr, &a_host, M, K, lda)?;
    upload_matrix(&dev, b_addr, &b_host, K, N, ldb)?;

    blas::sgemm(
        &dev, &kernel,
        M as u32, N as u32, K as u32,
        1.0, a_addr, lda,
             b_addr, ldb,
        0.0, c_addr, ldc,
        n_shires,
    )?;
    println!("sGEMM kernel returned (M={M}, N={N} [partial], K={K}).");

    let c_flat = download_matrix(&dev, c_addr, M, N, ldc)?;

    // Verify every element of C. All values are exact in f32 (K*(i+1)*(j+1)
    // <= 32*32*20 = 20480 < 2^15), so zero tolerance is appropriate.
    let mut n_checked = 0_usize;
    let mut n_errors  = 0_usize;
    for i in 0..M {
        for j in 0..N {
            let reference = K as f32 * (i + 1) as f32 * (j + 1) as f32;
            let device    = c_flat[i * N + j];
            let err       = (reference - device).abs();
            if err > 0.0 {
                eprintln!(
                    "FAIL C[{i}][{j}]: reference={reference}, device={device}, \
                     |err|={err:.2e} (tile col {})",
                    j / 16,
                );
                n_errors += 1;
            }
            n_checked += 1;
        }
    }

    // Print a representative element from each tile column for confirmation.
    for (label, i, j) in [
        ("full tile,  C[3][7]",   3_usize, 7_usize),   // tile col 0, cols 0..16
        ("partial tile, C[0][16]", 0,       16),         // tile col 1, first col
        ("partial tile, C[3][19]", 3,       19),         // tile col 1, last col
        ("partial tile, C[15][19]", 15,     19),         // tile col 1, bottom-right
    ] {
        let reference = K as f32 * (i + 1) as f32 * (j + 1) as f32;
        let device    = c_flat[i * N + j];
        let err       = (reference - device).abs();
        println!("{label}: reference={reference:.1}, device={device:.1}, |err|={err:.2e}");
    }

    if n_errors == 0 {
        println!("All {n_checked} elements correct -- partial-N verified on hardware.");
        Ok(())
    } else {
        Err(Error::Limit(format!(
            "{n_errors}/{n_checked} elements failed the partial-N check"
        )))
    }
}

/// Upload a row-major matrix to device memory with stride padding.
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
        let src_row = &data[r * cols..(r + 1) * cols];
        let src_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(src_row.as_ptr() as *const u8, cols * 4)
        };
        dev.memcpy_h2d(src_bytes, addr + r as u64 * row_bytes)?;
    }
    Ok(())
}

/// Download a strided matrix into a contiguous host Vec of `rows * cols` f32.
fn download_matrix(
    dev:  &Device<et_soc1::transport::IoctlTransport>,
    addr: u64,
    rows: usize,
    cols: usize,
    ldc:  u32,
) -> Result<Vec<f32>> {
    let mut out = vec![0.0f32; rows * cols];
    let mut raw = vec![0u8; ldc as usize];
    for r in 0..rows {
        dev.memcpy_d2h(addr + r as u64 * ldc as u64, &mut raw)?;
        for c in 0..cols {
            out[r * cols + c] = f32::from_le_bytes(
                raw[c * 4..c * 4 + 4].try_into().unwrap()
            );
        }
    }
    Ok(out)
}
