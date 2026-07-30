//! Host driver for the data-parallel reduction kernel (`et-k-rs` -> `reduce-rs`).
//!
//! Uploads a large array, launches the reduction across a shire of harts (each
//! reduces its disjoint slice into its own cache-line-padded partial), DMAs the
//! partials back, and combines them -- verifying the total against the known sum.
//!
//! Emulator (no hardware):
//! ```text
//! cargo run --features emu --example reduce -- \
//!     et-k-rs/target/riscv64imac-unknown-none-elf/release/reduce-rs
//! ```
//! Real hardware:
//! ```text
//! cargo run --example reduce -- \
//!     et-k-rs/target/riscv64imac-unknown-none-elf/release/reduce-rs
//! ```

use std::process::ExitCode;

use et_abi::ReduceArgs;
use et_soc1::trace::{DecodedEntry, TraceBuffer};
use et_soc1::{Device, TraceConfig};

/// Input length (elements). A multiple of `harts_per_shire * 16` keeps each
/// hart's slice cache-line aligned. 2^18 elements = 1 MiB.
const N: u32 = 1 << 18;
const TRACE_BUFFER_SIZE: u64 = 4096 * 2048;

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
        eprintln!("usage: reduce <reduce-rs.elf>");
        std::process::exit(2);
    });
    let elf = std::fs::read(&kernel_path).map_err(|e| et_soc1::Error::Io {
        op: "read kernel ELF",
        source: e,
    })?;

    let device = open_device()?;
    let di = device.dram_info();
    println!(
        "Device ready: DRAM base {:#x}, size {} bytes",
        di.base, di.size
    );

    // Size the launch to the device: one shire, all its harts. No hard-coded 64.
    let topo = device.topology()?;
    let shire_mask = topo.first_shire();
    let n_harts = topo.harts_per_shire;
    println!(
        "Topology: {} shire(s) present (mask {:#x}), {} harts/shire; launching on shire mask {:#x}",
        topo.num_shires(),
        topo.shire_mask,
        topo.harts_per_shire,
        shire_mask
    );

    let kernel = device.load_kernel(&elf)?;

    // Host input: element i = i + 1, so the exact sum is known.
    let host_in: Vec<u32> = (0..N).map(|i| i + 1).collect();

    // Typed device buffers: upload the input, allocate one cache-line-padded
    // partial per hart (the padding prevents false sharing). No byte casts, no
    // raw addresses at the call site.
    let input = device.upload(&host_in)?;
    let partials = device.alloc_padded::<u64>(n_harts as usize)?;
    let trace_buf = device.alloc(TRACE_BUFFER_SIZE)?;

    // Kernel args: the same struct the kernel reads (et-abi), so host and device
    // cannot drift on layout.
    let args = ReduceArgs {
        input: input.addr(),
        out: partials.addr(),
        n: N,
        n_harts,
    };

    // One typed call bundles the shire mask, the argument staging, and tracing.
    let launch = device.launch_spmd_traced(
        &kernel,
        shire_mask,
        &args,
        TraceConfig::full(trace_buf, shire_mask),
    );

    // Print any trace lines regardless of launch outcome (on an exception the
    // firmware still fills the trace buffer -- useful for diagnostics).
    if let Ok(tb) = TraceBuffer::parse(&out_trace(&device, trace_buf)?) {
        for entry in tb.entries() {
            if let DecodedEntry::String(s) = entry.decoded() {
                println!("[hart {}] {}", entry.hart_id, s.trim_end());
            }
        }
    }

    match &launch {
        Ok(r) => println!(
            "Kernel completed in {} cycles (waited {}).",
            r.timing.execute_dur, r.timing.wait_dur
        ),
        Err(e) => println!("Launch error: {e}"),
    }
    launch?;

    // Combine the per-hart partials (download extracts one u64 per cache line).
    let parts = device.download_padded(&partials)?;
    let total: u64 = parts.iter().copied().fold(0u64, u64::wrapping_add);
    let nonzero = parts.iter().filter(|&&p| p != 0).count();

    let expected = (N as u64) * (N as u64 + 1) / 2;
    println!(
        "\nReduction over {N} elements on {n_harts} harts ({nonzero} contributed):\n  \
         device total = {total}\n  expected     = {expected}"
    );
    if total == expected {
        println!("RESULT PASS");
        Ok(())
    } else {
        Err(et_soc1::Error::Protocol(format!(
            "reduction mismatch: got {total}, expected {expected}"
        )))
    }
}

/// DMA the trace buffer back for decoding.
fn out_trace<T: et_soc1::Transport>(
    device: &Device<T>,
    trace_buf: et_soc1::DeviceRegion,
) -> et_soc1::Result<Vec<u8>> {
    let mut host = vec![0u8; trace_buf.size as usize];
    device.memcpy_d2h(trace_buf.addr, &mut host)?;
    Ok(host)
}

#[cfg(feature = "emu")]
fn open_device() -> et_soc1::Result<Device<et_soc1::FfiTransport>> {
    let sdk_prefix = std::env::var("ET_SDK_PREFIX").unwrap_or_else(|_| "/opt/et".to_string());
    let run_dir = std::env::current_dir()
        .map_err(|e| et_soc1::Error::Io {
            op: "current_dir",
            source: e,
        })?
        .join("sysemu-run");
    std::fs::create_dir_all(&run_dir).map_err(|e| et_soc1::Error::Io {
        op: "create run dir",
        source: e,
    })?;
    eprintln!("Booting software emulator (this can take a while)...");
    Device::open_emulator(&sdk_prefix, &run_dir)
}

#[cfg(not(feature = "emu"))]
fn open_device() -> et_soc1::Result<Device> {
    Device::open(0)
}
