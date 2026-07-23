//! Pure-Rust re-implementation of the SDK `et-testdrive` "hello world".
//!
//! It opens an ET-SoC-1 device, loads a compute-kernel ELF, launches it on a
//! single shire with full user tracing, copies the trace buffer back over DMA
//! and prints every decoded string entry.
//!
//! This talks to the PCIe kernel driver directly and therefore requires real
//! hardware with `/dev/et0_ops` present; it cannot run against the software
//! emulator (see the crate-level documentation).
//!
//! Usage:
//!
//! ```text
//! cargo run --example hello -- /path/to/hello.elf
//! ```

use std::process::ExitCode;

use et_soc1::trace::{DecodedEntry, TraceBuffer};
use et_soc1::{Device, LaunchOptions, TraceConfig};

/// Single shire, matching the test drive's `kShireMask`.
const SHIRE_MASK: u64 = 0x1;
/// 8 MiB trace buffer, generously oversized for a few kernel prints.
const TRACE_BUFFER_SIZE: u64 = 4096 * 2048;

fn main() -> ExitCode {
    match run() {
        Ok(count) => {
            println!("Decoded {count} trace string entries.");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> et_soc1::Result<usize> {
    let kernel_path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: hello <kernel.elf>");
        std::process::exit(2);
    });

    let elf = std::fs::read(&kernel_path).map_err(|e| et_soc1::Error::Io {
        op: "read kernel ELF",
        source: e,
    })?;

    // Open device 0 and discover its DRAM geometry.
    let device = Device::open(0)?;
    let di = device.dram_info();
    println!(
        "Opened device 0: DRAM base {:#x}, size {} bytes",
        di.base, di.size
    );
    println!(
        "  DMA max_elem={} max_nodes={} dma_alignment={} (alignment={} bytes)",
        di.dma_max_elem_size,
        di.dma_max_elem_count,
        di.dma_alignment,
        di.alignment()
    );

    // Load the kernel and reserve a device-resident trace buffer.
    let kernel = device.load_kernel(&elf)?;
    let trace_buf = device.alloc(TRACE_BUFFER_SIZE)?;

    // Launch on one shire with a barrier and full user tracing.
    let opts = LaunchOptions::new(SHIRE_MASK)
        .with_trace(TraceConfig::full(trace_buf, SHIRE_MASK))
        .with_args(vec![0u8; 64]);
    let result = device.launch(&kernel, &opts)?;
    println!(
        "Kernel completed in {} cycles (waited {}).",
        result.timing.execute_dur, result.timing.wait_dur
    );

    // Copy the trace buffer back and decode it.
    let mut host_trace = vec![0u8; TRACE_BUFFER_SIZE as usize];
    device.memcpy_d2h(trace_buf.addr, &mut host_trace)?;

    let mut count = 0usize;
    match TraceBuffer::parse(&host_trace) {
        Ok(tb) => {
            for entry in tb.entries() {
                if let DecodedEntry::String(s) = entry.decoded() {
                    // The kernel's string already ends in a newline; trim it so
                    // the line is not double-spaced (the decoder stays faithful
                    // to the raw payload).
                    println!("[hart {}] {}", entry.hart_id, s.trim_end());
                    count += 1;
                }
            }
        }
        Err(e) => eprintln!("trace buffer not decodable: {e}"),
    }
    Ok(count)
}
