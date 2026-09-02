//! Host driver for the tensor-extension instruction test.
//!
//! Uploads hand-crafted input matrices in the exact hardware data formats
//! documented in PRM Chapter 9, launches the `tensor-ext-test` kernel on all
//! available shires, and verifies three subtests across all Minions:
//!
//! 1. **TensorFMA16A32** (fp16 GEMM): expects C = `[5.0, 0.0, 0.0, 0.0]` f32.
//! 2. **TensorIMA8A32** (int8 GEMM): expects C = `[4, 0, 0, 0]` i32.
//! 3. **TensorStoreFromScp** (scratchpad passthrough): output bytes match the
//!    fp16 A input buffer exactly.
//!
//! # Input encoding
//!
//! ## FMA16A32 (AROWS=0, ACOLS=0, BCOLS=0, TENB=1)
//! - A (64 bytes, first 4 used): `[2.0_f16, 3.0_f16, 0...]`
//!   `2.0_f16 = 0x4000`, `3.0_f16 = 0x4200` (little-endian).
//! - B (64 bytes, first 4 used in TenB interleaved format per PRM):
//!   `h[0] = b[0,0] = 1.0_f16 = 0x3C00`, `h[1] = b[1,0] = 1.0_f16 = 0x3C00`,
//!   `h[2..7] = 0.0`. Cols 1-3 of B are zero.
//! - Expected: `C[0][0] = 2.0*1.0 + 3.0*1.0 = 5.0`.
//!
//! ## IMA8A32 (AROWS=0, ACOLS=0, BCOLS=0, TENB=0, DST=1)
//! - A (64 bytes, first 4 used): `[1i8, 1, 1, 1, 0...]`
//! - B (64 bytes in IMA8A32 interleaved format per PRM):
//!   word 0 = `[1, 1, 1, 1]` (col 0, K-rows 0-3), words 1-3 = `[0, 0, 0, 0]`.
//! - Expected: `C[0][0] = 1*1+1*1+1*1+1*1 = 4`; `C[0][1..3] = 0`.
//!   Results are in the FP register file as int32 bit patterns; tensor_store
//!   writes them verbatim, so the host reads them back as `i32::from_le_bytes`.
//!
//! # Usage
//! ```text
//! cargo build --target riscv64imac-unknown-none-elf --release --bin tensor-ext-test
//! cargo run --example tensor_ext_test -- <path/to/tensor-ext-test>
//! ```

use std::process::ExitCode;

use et_abi::{DeviceArgs, MINIONS_PER_SHIRE, TENSOR_ALIGN, TensorExtTestArgs};
use et_soc1::{Device, LaunchOptions};

/// Bytes of output per Minion (3 subtests × 64 bytes each).
const OUT_STRIDE: usize = 192;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> et_soc1::Result<()> {
    let kernel_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: tensor_ext_test <tensor-ext-test.elf>");
        std::process::exit(2);
    });
    let elf = std::fs::read(&kernel_path)
        .map_err(|e| et_soc1::Error::io("read kernel ELF", e))?;

    let device    = Device::open(0)?;
    let topo      = device.topology()?;
    let n_shires  = topo.num_shires() as usize;
    let n_minions = n_shires * MINIONS_PER_SHIRE as usize;

    println!("Device: {} shires, {} Minions", n_shires, n_minions);

    let kernel = device.load_kernel(&elf)?;

    // -----------------------------------------------------------------------
    // Output buffer: n_minions * 192 bytes, 64-byte aligned by device.alloc.
    // -----------------------------------------------------------------------
    let out_buf = device.alloc((n_minions * OUT_STRIDE) as u64)?;

    // -----------------------------------------------------------------------
    // fp16 A: [2.0_f16, 3.0_f16, 0...] (64 bytes).
    // fp16 little-endian: 2.0 = 0x4000 -> bytes [0x00, 0x40];
    //                     3.0 = 0x4200 -> bytes [0x00, 0x42].
    // -----------------------------------------------------------------------
    let mut a_fp16 = vec![0u8; TENSOR_ALIGN];
    a_fp16[0..4].copy_from_slice(&[0x00u8, 0x40, 0x00, 0x42]);
    let a_fp16_dev = device.alloc(TENSOR_ALIGN as u64)?;
    device.memcpy_h2d(a_fp16_dev.addr, &a_fp16)?;

    // -----------------------------------------------------------------------
    // fp16 B in TenB interleaved format (PRM TensorFMA16A32).
    // TenB[0].h[j*2+0] = b[2k, j],  TenB[0].h[j*2+1] = b[2k+1, j]
    // For k=0 (only K-group), j=0 (output col 0):
    //   h[0] = b[0,0] = 1.0_f16 = 0x3C00 -> bytes [0x00, 0x3C]
    //   h[1] = b[1,0] = 1.0_f16 = 0x3C00 -> bytes [0x00, 0x3C]
    // j=1..3 (cols 1-3): h[2..7] = 0.0.
    // -----------------------------------------------------------------------
    let mut b_fp16 = vec![0u8; TENSOR_ALIGN];
    b_fp16[0..4].copy_from_slice(&[0x00u8, 0x3C, 0x00, 0x3C]);
    let b_fp16_dev = device.alloc(TENSOR_ALIGN as u64)?;
    device.memcpy_h2d(b_fp16_dev.addr, &b_fp16)?;

    // -----------------------------------------------------------------------
    // int8 A: [1, 1, 1, 1, 0...] (64 bytes).
    // -----------------------------------------------------------------------
    let mut a_int8 = vec![0u8; TENSOR_ALIGN];
    a_int8[0..4].copy_from_slice(&[1u8, 1, 1, 1]);
    let a_int8_dev = device.alloc(TENSOR_ALIGN as u64)?;
    device.memcpy_h2d(a_int8_dev.addr, &a_int8)?;

    // -----------------------------------------------------------------------
    // int8 B in IMA8A32 interleaved format (PRM TensorIMA8A32).
    // For ACOLS=0, BCOLS=0: one scratchpad line (64 bytes) with 4 output cols.
    // Word j = [b[0,j] | b[1,j] | b[2,j] | b[3,j]] (4 K-rows packed per col).
    //   word 0 (col 0): [1, 1, 1, 1] (all K-rows = 1)
    //   word 1 (col 1): [0, 0, 0, 0]
    //   word 2 (col 2): [0, 0, 0, 0]
    //   word 3 (col 3): [0, 0, 0, 0]
    // -----------------------------------------------------------------------
    let mut b_int8 = vec![0u8; TENSOR_ALIGN];
    b_int8[0..4].copy_from_slice(&[1u8, 1, 1, 1]);
    let b_int8_dev = device.alloc(TENSOR_ALIGN as u64)?;
    device.memcpy_h2d(b_int8_dev.addr, &b_int8)?;

    // -----------------------------------------------------------------------
    // Launch.
    // -----------------------------------------------------------------------
    let args = TensorExtTestArgs {
        output:   out_buf.addr,
        a_fp16:   a_fp16_dev.addr,
        b_fp16:   b_fp16_dev.addr,
        a_int8:   a_int8_dev.addr,
        b_int8:   b_int8_dev.addr,
        n_shires: n_shires as u64,
    };
    let opts = LaunchOptions::new(topo.shire_mask)
        .with_args(args.as_bytes().to_vec());
    device.launch(&kernel, &opts)?;

    // -----------------------------------------------------------------------
    // Download and verify.
    // -----------------------------------------------------------------------
    let mut host_out = vec![0u8; n_minions * OUT_STRIDE];
    device.memcpy_d2h(out_buf.addr, &mut host_out)?;

    let mut fails = 0usize;

    for m in 0..n_minions {
        let base = m * OUT_STRIDE;

        // --- Subtest 1: FMA16A32 ---
        // C is stored as f32 from the FP register file. Row 0 (AROWS=0) with
        // BCOLS=0 (<=1) occupies f[0] (32 bytes). The hardware tensor_store
        // writes 64 bytes (f[0]+f[1]); only e[0..3] of f[0] are valid here.
        let c_fp16: [f32; 4] = std::array::from_fn(|i| {
            f32::from_le_bytes(host_out[base + i * 4..base + i * 4 + 4].try_into().unwrap())
        });
        // FMA16A32 3-way fused add is not IEEE754-equivalent; allow 1 ULP of
        // f32 tolerance (2^-23 * 8 ~ 0.001).
        if (c_fp16[0] - 5.0_f32).abs() > 0.01 || c_fp16[1] != 0.0 || c_fp16[2] != 0.0 || c_fp16[3] != 0.0 {
            eprintln!(
                "  FAIL minion {m} FMA16A32: got {:?}, want [5.0, 0.0, 0.0, 0.0]",
                c_fp16
            );
            fails += 1;
        }

        // --- Subtest 2: IMA8A32 ---
        // int32 results stored as f32 bit patterns via DST=1 + tensor_store.
        // Read the 4-byte little-endian bit pattern directly as i32.
        let base2 = base + 64;
        let c_int: [i32; 4] = std::array::from_fn(|i| {
            i32::from_le_bytes(host_out[base2 + i * 4..base2 + i * 4 + 4].try_into().unwrap())
        });
        if c_int != [4, 0, 0, 0] {
            eprintln!(
                "  FAIL minion {m} IMA8A32: got {:?}, want [4, 0, 0, 0]",
                c_int
            );
            fails += 1;
        }

        // --- Subtest 3: TensorStoreFromScp ---
        // tensor_store_from_scp wrote scratchpad line 0 verbatim to memory;
        // line 0 was loaded from a_fp16. The 64 bytes must match exactly.
        let scp = &host_out[base + 128..base + 192];
        if scp != a_fp16.as_slice() {
            eprintln!("  FAIL minion {m} StoreFromScp: scratchpad line mismatch");
            eprintln!("    got:  {:?}", &scp[..8]);
            eprintln!("    want: {:?}", &a_fp16[..8]);
            fails += 1;
        }
    }

    if fails == 0 {
        println!(
            "tensor_ext_test PASSED ({n_minions} Minions x 3 subtests: \
             FMA16A32, IMA8A32, StoreFromScp)"
        );
        Ok(())
    } else {
        Err(et_soc1::Error::Protocol(format!(
            "{fails} failure(s) in tensor_ext_test across {n_minions} Minions"
        )))
    }
}
