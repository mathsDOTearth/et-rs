//! The "hello world" flow driven against the SDK **software emulator**, for
//! hosts without ET-SoC-1 hardware. Requires the `emu` feature:
//!
//! ```text
//! cargo run --features emu --example hello_sysemu -- /path/to/hello.elf
//! ```
//!
//! It uses exactly the same `Device` API as the hardware example; only the
//! transport differs (`Device::open_emulator` instead of `Device::open`). The
//! emulator boots firmware on startup, which takes appreciable time.

use std::process::ExitCode;

use et_soc1::trace::{DecodedEntry, TraceBuffer};
use et_soc1::{Device, LaunchOptions, TraceConfig};

const SHIRE_MASK: u64 = 0x1;
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
        eprintln!("usage: hello_sysemu <kernel.elf>");
        std::process::exit(2);
    });
    let sdk_prefix = std::env::var("ET_SDK_PREFIX").unwrap_or_else(|_| "/opt/et".to_string());

    let elf = std::fs::read(&kernel_path).map_err(|e| et_soc1::Error::Io {
        op: "read kernel ELF",
        source: e,
    })?;

    // A writable directory for the emulator's log files.
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
    let device = Device::open_emulator(&sdk_prefix, &run_dir)?;
    let di = device.dram_info();
    println!(
        "Emulator ready: DRAM base {:#x}, size {} bytes; DMA max_elem={} max_nodes={} alignment={}",
        di.base,
        di.size,
        di.dma_max_elem_size,
        di.dma_max_elem_count,
        di.alignment()
    );

    let kernel = device.load_kernel(&elf)?;
    let trace_buf = device.alloc(TRACE_BUFFER_SIZE)?;

    let opts = LaunchOptions::new(SHIRE_MASK)
        .with_trace(TraceConfig::full(trace_buf, SHIRE_MASK))
        .with_args(vec![0u8; 64]);
    let result = device.launch(&kernel, &opts)?;
    println!(
        "Kernel completed in {} cycles (waited {}).",
        result.timing.execute_dur, result.timing.wait_dur
    );

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
