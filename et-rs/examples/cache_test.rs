//! Cache-coherence test: verifies that `cache_writeback` makes Minion-written
//! data visible to host DMA, and instruments the launch to diagnose faults.
//!
//! Each primary Minion hart writes its global Minion index into its own
//! cache-line-padded output cell and calls `cache_writeback` before `fence`.
//! The host downloads the output array and asserts every cell holds the
//! expected Minion index.
//!
//! # Known intermittent fault
//!
//! On the current card build this kernel launch intermittently fails with
//! EXCEPTION (status 2) once roughly 16 or more shires participate, at a measured
//! rate of 50-90 percent, with a non-deterministic partial faulting-shire mask.
//! The fault is isolated to the U-mode `flush_va` cache op: it is the only kernel
//! that issues one, and non-cache-op kernels (sgemm, reduce, tensor) run reliably
//! at the same scale. It is independent of the writeback destination, the launch
//! BARRIER flag, the exception buffer, and the pre-writeback fence (all tested and
//! ruled out; see the diagnostic controls below).
//!
//! The fault is NOT caused by the kernel binary itself, nor by the instruction gap
//! between `csrw flush_va` and `csrwi tensor_wait`: both hypotheses were tested
//! on hardware by running the Rust kernel ELF and a 5-NOP-gap C kernel variant
//! through a C++ host program, each passing 20/20 at 32 shires. The trigger is
//! therefore in the Rust host path. An allocation-order bug (output buffer
//! allocated before kernel load) has been fixed in this release; whether that
//! resolves the EXCEPTION is pending hardware confirmation. The emulator cannot
//! reproduce this fault because it suppresses `flush_va` side-effects.
//!
//! # Diagnostic controls
//!
//! An optional shire count narrows a concurrency-dependent fault (run on 1 shire
//! versus all 32); an optional destination level narrows a DDR-specific fault
//! (flush to L2/L3 instead of Mem).
//!
//! `ET_NO_BARRIER=1` clears the launch BARRIER flag. Tested and refuted: the
//! failure rate is unchanged with the barrier on or off, so the firmware
//! barrier/drain path is not the trigger. Retained as a documented control.
//!
//! `ET_EXC_BUFFER=1` additionally supplies a U-mode exception buffer and, on a
//! launch exception, decodes the execution context the firmware leaves there
//! (`mcause`, `mepc`, `mtval`), distinguishing an illegal instruction (cache-op
//! feature gate) from a page/access fault (the flush path). Off by default: on the
//! current card firmware a non-zero exception buffer is itself rejected with
//! EXCEPTION (status 2) regardless of kernel correctness, and firmware writes
//! nothing into it, so it is reserved for probing a firmware build that supports
//! the feature.
//!
//! # Usage
//! ```text
//! cargo run --example cache_test -- <cache-test-rs.elf> [shires] [dest]
//! ```
//! `shires` defaults to all present; `dest` is 1 = L2, 2 = L3, 3 = Mem (default,
//! the only host-visible level).

use std::process::ExitCode;

use et_abi::{CacheTestArgs, DeviceArgs, MINIONS_PER_SHIRE};
use et_soc1::{Device, LaunchOptions};

/// Size of the exception buffer: the firmware writes an `execution_context_t`
/// (type, cycles, hart_id, sepc, sstatus, stval, scause, user_error, 31 GPRs =
/// 312 bytes) here on a U-mode trap; 512 bytes gives margin and 64-byte alignment.
const EXC_BUF_LEN: usize = 512;

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
    let argv: Vec<String> = std::env::args().collect();
    let kernel_path = argv.get(1).cloned().unwrap_or_else(|| {
        eprintln!("usage: cache_test <cache-test-rs.elf> [shires] [dest:1=L2,2=L3,3=Mem]");
        std::process::exit(2);
    });
    let elf = std::fs::read(&kernel_path).map_err(|e| et_soc1::Error::io("read kernel ELF", e))?;

    let device = Device::open(0)?;
    let topo = device.topology()?;
    let max_shires = topo.num_shires() as usize;

    // Narrowing controls. `shires` (default all) isolates concurrency-dependent
    // faults; `dest` (default 3 = Mem) isolates the DDR writeback path.
    let n_shires = argv
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .map(|k| k.clamp(1, max_shires))
        .unwrap_or(max_shires);
    let dest: u64 = argv.get(3).and_then(|s| s.parse().ok()).unwrap_or(3);
    let n_minions = n_shires * MINIONS_PER_SHIRE as usize;

    // Launch on the first `n_shires` present shires.
    let shire_mask = if n_shires >= max_shires {
        topo.shire_mask
    } else {
        (1u64 << n_shires) - 1
    };

    println!(
        "Device: {} shires present (mask {:#x}); running {} shire(s), {} Minions, dest = {}",
        max_shires,
        topo.shire_mask,
        n_shires,
        n_minions,
        dest_name(dest),
    );

    // Kernel must be loaded first so it lands at DRAM base (its link address).
    // Allocating output before load_kernel would place the output buffer at
    // DRAM base and kernel code would then DMA-overwrite it.
    let kernel = device.load_kernel(&elf)?;

    // Output: one u32 per Minion, each on its own 64-byte cache line.
    let output = device.alloc_padded::<u32>(n_minions)?;

    let args = CacheTestArgs {
        output: output.addr(),
        n_shires: n_shires as u64,
        dest,
    };

    // Opt-in exception-context capture. Setting a non-zero exception_buffer is a
    // firmware-supported feature only on some builds: on the current card build
    // it is itself rejected with EXCEPTION (status 2) regardless of kernel
    // correctness, and firmware leaves the buffer untouched. It is therefore off
    // by default (the launch then mirrors the passing double_buffer path) and
    // enabled only via ET_EXC_BUFFER=1 for probing a future firmware.
    let exc = if std::env::var_os("ET_EXC_BUFFER").is_some() {
        // Pre-zeroed so a non-fault run leaves recognisable zeros.
        let region = device.alloc(EXC_BUF_LEN as u64)?;
        device.memcpy_h2d(&[0u8; EXC_BUF_LEN], region.addr)?;
        println!(
            "ET_EXC_BUFFER set: capturing U-mode context @ {:#x}",
            region.addr
        );
        Some(region)
    } else {
        None
    };

    println!("Launching cache_writeback test ...");
    let mut opts = LaunchOptions::new(shire_mask).with_args(args.as_bytes().to_vec());
    // ET_NO_BARRIER clears the launch BARRIER flag (diagnostic). Tested and
    // refuted: a stress loop measured the same 50-90 percent EXCEPTION rate with
    // the barrier on or off, so the firmware barrier/drain path is not the fault
    // trigger. Retained so the comparison can be reproduced without recompiling.
    if std::env::var_os("ET_NO_BARRIER").is_some() {
        opts = opts.without_barrier();
    }
    if let Some(ref region) = exc {
        opts.exception_buffer = region.addr;
    }
    if let Err(e) = device.launch(&kernel, &opts) {
        // Decode the U-mode context the firmware saved before returning the error.
        if let Some(ref region) = exc {
            let mut raw = vec![0u8; EXC_BUF_LEN];
            device.memcpy_d2h(region.addr, &mut raw)?;
            print_exception(&raw);
        }
        return Err(e);
    }
    println!("Kernel returned.");

    // Download the per-Minion outputs (strips 64-byte padding).
    let results = device.download_padded(&output)?;

    let mut failures = 0_usize;
    for (i, &val) in results.iter().enumerate() {
        if val != i as u32 {
            eprintln!("  FAIL output[{i}] = {val:#010x}  (expected {i:#010x})");
            failures += 1;
        }
    }

    if failures == 0 {
        println!("cache_writeback PASSED: all {} cells correct", n_minions);
        Ok(())
    } else {
        Err(et_soc1::Error::Protocol(format!(
            "cache_writeback FAILED: {failures}/{n_minions} cells incorrect"
        )))
    }
}

fn dest_name(dest: u64) -> &'static str {
    match dest {
        1 => "L2",
        2 => "L3",
        _ => "Mem",
    }
}

/// Decode the `execution_context_t` the firmware writes into the exception buffer
/// on a U-mode trap (fields in declaration order, `packed, aligned(64)`).
fn print_exception(raw: &[u8]) {
    if raw.len() < 56 {
        eprintln!("  exception buffer too short to decode");
        return;
    }
    let rd = |o: usize| u64::from_le_bytes(raw[o..o + 8].try_into().unwrap());
    let ty = rd(0); // context type (0 if firmware wrote nothing)
    let hart_id = rd(16);
    let mepc = rd(24); // sepc: PC of the faulting instruction
    let mstatus = rd(32);
    let mtval = rd(40); // stval: faulting address or instruction bits
    let mcause = rd(48); // scause: trap cause

    if ty == 0 && mepc == 0 && mcause == 0 {
        eprintln!("  exception buffer empty: firmware saved no U-mode context here");
        return;
    }
    eprintln!("  U-mode exception context (first faulting hart):");
    eprintln!("    hart_id = {hart_id}");
    eprintln!("    mcause  = {mcause:#x}  ({})", mcause_name(mcause));
    eprintln!("    mepc    = {mepc:#x}  (faulting instruction PC)");
    eprintln!("    mtval   = {mtval:#x}  (faulting address / instruction bits)");
    eprintln!("    mstatus = {mstatus:#x}");
}

/// Human-readable RISC-V trap cause (synchronous exceptions only).
fn mcause_name(cause: u64) -> &'static str {
    if cause >> 63 != 0 {
        return "interrupt";
    }
    match cause & 0xff {
        0 => "instruction address misaligned",
        1 => "instruction access fault",
        2 => "illegal instruction",
        3 => "breakpoint",
        4 => "load address misaligned",
        5 => "load access fault",
        6 => "store/AMO address misaligned",
        7 => "store/AMO access fault",
        8 => "environment call from U-mode",
        12 => "instruction page fault",
        13 => "load page fault",
        15 => "store/AMO page fault",
        _ => "other/unknown",
    }
}
