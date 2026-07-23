//! The high-level [`Device`] handle: DRAM allocation, kernel loading, launch,
//! device-to-host DMA and trace extraction.
//!
//! [`Device`] is generic over a [`Transport`]; the default,
//! [`IoctlTransport`], drives real hardware through `/dev/etN_ops`. The device
//! command model is single-threaded, so state that mutates during otherwise
//! read-only operations (the DRAM bump pointer and the tag counter) is held in
//! `Cell`s and the command methods take `&self`.

use crate::elf;
use crate::error::{Error, Result};
use crate::ffi::ops;
use crate::proto::{self, cmd_flags, desc_flags};
use crate::transport::{DramInfo, IoctlTransport, PoppedResponse, Transport};
use std::cell::Cell;
use std::time::{Duration, Instant};

/// Default time to wait for a submission-queue slot or a completion response.
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);

/// A device-resident kernel, ready to be launched.
#[derive(Clone, Copy, Debug)]
pub struct LoadedKernel {
    /// Device address at which execution begins (the ELF entry point).
    pub code_start_address: u64,
}

/// A handle to a region of device DRAM returned by [`Device::alloc`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceRegion {
    /// Device physical base address of the region.
    pub addr: u64,
    /// Size of the region in bytes.
    pub size: u64,
}

impl DeviceRegion {
    /// The half-open address range `[addr, addr + size)`.
    pub fn end(&self) -> u64 {
        self.addr + self.size
    }
}

/// U-mode trace capture configuration for a kernel launch, mirroring
/// [`proto::TraceInitInfo`] with a device-resident trace buffer.
#[derive(Clone, Copy, Debug)]
pub struct TraceConfig {
    /// Device address of the trace buffer (typically an [`Device::alloc`] region).
    pub buffer: u64,
    /// Size of the trace buffer in bytes.
    pub buffer_size: u32,
    /// Per-hart free-space threshold at which the device raises a full event.
    pub threshold: u32,
    /// Bitmask of shires for which trace capture is enabled.
    pub shire_mask: u64,
    /// Bitmask of threads within a shire for which trace capture is enabled.
    pub thread_mask: u64,
    /// Bitmask selecting which events to trace.
    pub event_mask: u32,
    /// Bitmask selecting which filters apply to the traced events.
    pub filter_mask: u32,
}

impl TraceConfig {
    /// Enable full user tracing of every thread, event and filter for `shire_mask`,
    /// dumping into the whole of `buffer`. Mirrors the configuration used by the
    /// SDK "hello world" test drive.
    pub fn full(buffer: DeviceRegion, shire_mask: u64) -> Self {
        TraceConfig {
            buffer: buffer.addr,
            buffer_size: buffer.size as u32,
            threshold: 0,
            shire_mask,
            thread_mask: u64::MAX,
            event_mask: u32::MAX,
            filter_mask: u32::MAX,
        }
    }

    fn to_init_info(self) -> proto::TraceInitInfo {
        proto::TraceInitInfo {
            buffer: self.buffer,
            buffer_size: self.buffer_size,
            threshold: self.threshold,
            shire_mask: self.shire_mask,
            thread_mask: self.thread_mask,
            event_mask: self.event_mask,
            filter_mask: self.filter_mask,
        }
    }
}

/// Options controlling a single kernel launch.
#[derive(Clone, Debug)]
pub struct LaunchOptions {
    /// Bitmask of compute shires the kernel executes on.
    pub shire_mask: u64,
    /// Whether to drain outstanding commands before launching (barrier).
    pub barrier: bool,
    /// Whether to flush the L3 cache before launching.
    pub flush_l3: bool,
    /// Optional U-mode trace configuration.
    pub trace: Option<TraceConfig>,
    /// Optional embedded kernel arguments blob.
    pub args: Vec<u8>,
    /// Optional U-mode stack configuration.
    pub stack: Option<proto::UserStackCfg>,
    /// Device address of a U-mode exception buffer (0 if unused).
    pub exception_buffer: u64,
    /// Submission queue to push the launch command onto.
    pub sq_index: u16,
}

impl LaunchOptions {
    /// Launch on `shire_mask` with a barrier and no tracing or arguments.
    pub fn new(shire_mask: u64) -> Self {
        LaunchOptions {
            shire_mask,
            barrier: true,
            flush_l3: false,
            trace: None,
            args: Vec::new(),
            stack: None,
            exception_buffer: 0,
            sq_index: 0,
        }
    }

    /// Enable U-mode tracing with the given configuration.
    pub fn with_trace(mut self, trace: TraceConfig) -> Self {
        self.trace = Some(trace);
        self
    }

    /// Attach an embedded kernel-arguments blob.
    pub fn with_args(mut self, args: Vec<u8>) -> Self {
        self.args = args;
        self
    }
}

/// Timing counters reported alongside a kernel-launch response, in device cycles.
#[derive(Clone, Copy, Debug, Default)]
pub struct LaunchTiming {
    /// Timestamp at which the command was dispatched.
    pub start_ts: u64,
    /// Cycles between dispatch and completion.
    pub execute_dur: u64,
    /// Cycles between arrival and dispatch.
    pub wait_dur: u64,
}

/// Outcome of a successful [`Device::launch`].
#[derive(Clone, Copy, Debug, Default)]
pub struct LaunchResult {
    /// Device timing counters for the launch.
    pub timing: LaunchTiming,
}

/// A connected ET-SoC-1 device.
pub struct Device<T: Transport = IoctlTransport> {
    transport: T,
    dram: DramInfo,
    /// Bump-allocation cursor within the user DRAM region.
    next: Cell<u64>,
    /// Monotonic command correlation tag.
    tag: Cell<u16>,
}

impl Device<IoctlTransport> {
    /// Open device `index` (`/dev/et{index}_ops`) and query its DRAM geometry.
    pub fn open(index: u32) -> Result<Self> {
        Self::with_transport(IoctlTransport::open(index)?)
    }
}

#[cfg(feature = "emu")]
impl Device<crate::transport::FfiTransport> {
    /// Boot the SDK software emulator and open it as a device.
    ///
    /// `sdk_prefix` is the SDK install root (e.g. `/opt/et`) and `run_dir` a
    /// writable directory for emulator logs. This blocks while the emulator
    /// boots firmware. Requires the `emu` feature.
    pub fn open_emulator<P: AsRef<std::path::Path>, Q: AsRef<std::path::Path>>(
        sdk_prefix: P,
        run_dir: Q,
    ) -> Result<Self> {
        Self::with_transport(crate::transport::FfiTransport::new(sdk_prefix, run_dir)?)
    }
}

impl<T: Transport> Device<T> {
    /// Build a device over an explicit transport (an alternative backend, or a
    /// test double). The DRAM geometry is queried immediately.
    pub fn with_transport(transport: T) -> Result<Self> {
        let dram = transport.dram_info()?;
        Ok(Device {
            transport,
            dram,
            next: Cell::new(dram.base),
            tag: Cell::new(0),
        })
    }

    /// The device's user DRAM region geometry and DMA limits.
    pub fn dram_info(&self) -> DramInfo {
        self.dram
    }

    /// Borrow the underlying transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Bytes still available in the DRAM bump allocator.
    pub fn dram_available(&self) -> u64 {
        (self.dram.base + self.dram.size).saturating_sub(self.next.get())
    }

    /// Allocate a naturally aligned region of device DRAM.
    ///
    /// This is a monotonic bump allocator: regions are never individually freed.
    /// Alignment follows the device's advertised requirement
    /// ([`DramInfo::alignment`]).
    pub fn alloc(&self, size: u64) -> Result<DeviceRegion> {
        let align = self.dram.alignment().max(1);
        let start = align_up(self.next.get(), align);
        let end = self.dram.base + self.dram.size;
        if start > end || size > end - start {
            return Err(Error::OutOfMemory {
                requested: size,
                available: self.dram_available(),
            });
        }
        self.next.set(start + size);
        Ok(DeviceRegion { addr: start, size })
    }

    /// Load a RISC-V ELF device kernel into device DRAM.
    ///
    /// Compute kernels are position-dependent and linked at a fixed U-mode
    /// address that coincides with the base of the user DRAM region; there is no
    /// firmware-side ELF loader for them (that is what `FW_UPDATE` is for, and it
    /// rejects kernel ELFs). Loading therefore DMA-writes each `PT_LOAD` segment
    /// to its virtual address, zero-filling any `.bss` tail, and reserves the
    /// occupied DRAM so subsequent [`Device::alloc`] calls do not overlap the
    /// code. The returned [`LoadedKernel`] carries the ELF entry point for use as
    /// the launch `code_start_address`.
    ///
    /// Call this before allocating other regions so the kernel lands at the DRAM
    /// base, matching its link address.
    pub fn load_kernel(&self, elf_image: &[u8]) -> Result<LoadedKernel> {
        let image = elf::parse(elf_image)?;
        let region_end = self.dram.base + self.dram.size;
        let mut occupied_end = self.next.get();

        for seg in &image.segments {
            if seg.vaddr < self.dram.base || seg.vaddr + seg.mem_size > region_end {
                return Err(Error::Limit(format!(
                    "kernel segment [{:#x}, {:#x}) lies outside the DRAM region [{:#x}, {:#x})",
                    seg.vaddr,
                    seg.vaddr + seg.mem_size,
                    self.dram.base,
                    region_end
                )));
            }
            if seg.file_size > 0 {
                let start = seg.file_offset as usize;
                let end = start + seg.file_size as usize;
                self.memcpy_h2d(&elf_image[start..end], seg.vaddr)?;
            }
            // Zero-initialise the `.bss` tail present in memory but not in the file.
            if seg.mem_size > seg.file_size {
                let zeros = vec![0u8; (seg.mem_size - seg.file_size) as usize];
                self.memcpy_h2d(&zeros, seg.vaddr + seg.file_size)?;
            }
            occupied_end = occupied_end.max(seg.vaddr + seg.mem_size);
        }

        // Reserve the DRAM the kernel occupies against future allocations.
        let align = self.dram.alignment().max(1);
        self.next.set(align_up(occupied_end, align).min(region_end));

        Ok(LoadedKernel {
            code_start_address: image.entry,
        })
    }

    /// Update device firmware via `FW_UPDATE`.
    ///
    /// This is for signed firmware images, not compute kernels; use
    /// [`Device::load_kernel`] for the latter.
    pub fn update_firmware(&self, image: &[u8]) -> Result<()> {
        self.transport.fw_update(image)
    }

    /// Launch a loaded kernel and wait for its completion response.
    pub fn launch(&self, kernel: &LoadedKernel, opts: &LaunchOptions) -> Result<LaunchResult> {
        let mut flags: u16 = 0;
        if opts.barrier {
            flags |= cmd_flags::BARRIER;
        }
        if opts.flush_l3 {
            flags |= cmd_flags::FLUSH_L3;
        }

        // Argument payload layout, per device_ops_kernel_launch_cmd_t: trace
        // configuration (40 B), then stack configuration (8 B), then embedded
        // kernel arguments, each present only when its flag is set.
        let mut payload = Vec::new();
        if let Some(trace) = opts.trace {
            flags |= cmd_flags::COMPUTE_KERNEL_TRACE;
            payload.extend_from_slice(&trace.to_init_info().to_bytes());
        }
        if let Some(stack) = opts.stack {
            flags |= cmd_flags::USER_STACK_CFG;
            payload.extend_from_slice(&stack.to_bytes());
        }
        if !opts.args.is_empty() {
            if opts.args.len() > ops::DEVICE_OPS_KERNEL_LAUNCH_ARGS_PAYLOAD_MAX as usize {
                return Err(Error::Limit(format!(
                    "kernel args {} exceed maximum {}",
                    opts.args.len(),
                    ops::DEVICE_OPS_KERNEL_LAUNCH_ARGS_PAYLOAD_MAX
                )));
            }
            flags |= cmd_flags::KERNEL_ARGS_EMBEDDED;
            payload.extend_from_slice(&opts.args);
        }

        let tag = self.next_tag();
        let cmd = proto::build_kernel_launch(
            tag,
            flags,
            kernel.code_start_address,
            0, // arguments are embedded in the payload, not referenced by pointer
            opts.exception_buffer,
            opts.shire_mask,
            &payload,
        );

        let rsp = self.submit(opts.sq_index, &cmd, 0, tag)?;
        let status = proto::response_status(&rsp.bytes)
            .ok_or_else(|| Error::Protocol("kernel-launch response truncated".into()))?;
        if status != ops::DEV_OPS_API_KERNEL_LAUNCH_RESPONSE::DEV_OPS_API_KERNEL_LAUNCH_RESPONSE_KERNEL_COMPLETED {
            return Err(Error::Device {
                command: "kernel-launch",
                code: status,
            });
        }
        Ok(LaunchResult {
            timing: parse_launch_timing(&rsp.bytes),
        })
    }

    /// Copy `dst.len()` bytes from device address `src` into host memory via a
    /// DMA read-list command, splitting the transfer to honour the device's DMA
    /// element-size and element-count limits.
    ///
    /// The transfer is staged through a transport-provided DMA host buffer (see
    /// [`crate::transport::DmaHostBuffer`]) and copied out afterwards, so it works
    /// whether the backend pins arbitrary host memory or requires registered DMA
    /// memory.
    pub fn memcpy_d2h(&self, src: u64, dst: &mut [u8]) -> Result<()> {
        let total = dst.len();
        if total == 0 {
            return Ok(());
        }
        let max_elem = (self.dram.dma_max_elem_size as usize).max(1);
        let max_nodes = (self.dram.dma_max_elem_count as usize).max(1);

        let host = self.transport.dma_host_buffer(total)?;
        let hvirt = host.virt_addr();
        let hphys = host.phys_addr();

        let mut offset = 0usize;
        let mut nodes: Vec<proto::DmaReadNode> = Vec::with_capacity(max_nodes);
        while offset < total {
            let len = (total - offset).min(max_elem);
            nodes.push(proto::DmaReadNode {
                dst_host_virt_addr: hvirt + offset as u64,
                dst_host_phy_addr: node_phys(hphys, offset),
                src_device_phy_addr: src + offset as u64,
                size: len as u32,
                _pad: [0; 4],
            });
            offset += len;
            if nodes.len() == max_nodes || offset >= total {
                self.dma_read_command(&nodes)?;
                nodes.clear();
            }
        }
        dst.copy_from_slice(host.as_slice());
        Ok(())
    }

    /// Copy `src.len()` bytes from host memory to device address `dst` via a DMA
    /// write-list command, splitting the transfer to honour the device's DMA
    /// element-size and element-count limits. The data is staged through a
    /// transport-provided DMA host buffer.
    pub fn memcpy_h2d(&self, src: &[u8], dst: u64) -> Result<()> {
        let total = src.len();
        if total == 0 {
            return Ok(());
        }
        let max_elem = (self.dram.dma_max_elem_size as usize).max(1);
        let max_nodes = (self.dram.dma_max_elem_count as usize).max(1);

        let mut host = self.transport.dma_host_buffer(total)?;
        host.as_mut_slice().copy_from_slice(src);
        let hvirt = host.virt_addr();
        let hphys = host.phys_addr();

        let mut offset = 0usize;
        let mut nodes: Vec<proto::DmaWriteNode> = Vec::with_capacity(max_nodes);
        while offset < total {
            let len = (total - offset).min(max_elem);
            nodes.push(proto::DmaWriteNode {
                src_host_virt_addr: hvirt + offset as u64,
                src_host_phy_addr: node_phys(hphys, offset),
                dst_device_phy_addr: dst + offset as u64,
                size: len as u32,
                _pad: [0; 4],
            });
            offset += len;
            if nodes.len() == max_nodes || offset >= total {
                self.dma_write_command(&nodes)?;
                nodes.clear();
            }
        }
        Ok(())
    }

    /// Extract the compute-minion trace buffer (`TRACE_BUFFER_CM`).
    pub fn extract_cm_trace(&self) -> Result<Vec<u8>> {
        self.extract_trace(ops::trace_buffer_type::TRACE_BUFFER_CM as u8)
    }

    /// Extract a device trace buffer of the given `trace_buffer_type`.
    pub fn extract_trace(&self, trace_type: u8) -> Result<Vec<u8>> {
        self.transport.extract_trace(trace_type)
    }

    // --- internals ---

    fn dma_read_command(&self, nodes: &[proto::DmaReadNode]) -> Result<()> {
        let tag = self.next_tag();
        let cmd = proto::build_dma_readlist(tag, cmd_flags::BARRIER, nodes);
        let rsp = self.submit(0, &cmd, desc_flags::DMA, tag)?;
        let status = proto::response_status(&rsp.bytes)
            .ok_or_else(|| Error::Protocol("DMA read-list response truncated".into()))?;
        if status != ops::DEV_OPS_API_DMA_RESPONSE::DEV_OPS_API_DMA_RESPONSE_COMPLETE {
            return Err(Error::Device {
                command: "dma-readlist",
                code: status,
            });
        }
        Ok(())
    }

    fn dma_write_command(&self, nodes: &[proto::DmaWriteNode]) -> Result<()> {
        let tag = self.next_tag();
        let cmd = proto::build_dma_writelist(tag, cmd_flags::BARRIER, nodes);
        let rsp = self.submit(0, &cmd, desc_flags::DMA, tag)?;
        let status = proto::response_status(&rsp.bytes)
            .ok_or_else(|| Error::Protocol("DMA write-list response truncated".into()))?;
        if status != ops::DEV_OPS_API_DMA_RESPONSE::DEV_OPS_API_DMA_RESPONSE_COMPLETE {
            return Err(Error::Device {
                command: "dma-writelist",
                code: status,
            });
        }
        Ok(())
    }

    /// Push a command and block for the response bearing `expected_tag`.
    fn submit(
        &self,
        sq_index: u16,
        cmd: &[u8],
        desc_flags: u8,
        expected_tag: u16,
    ) -> Result<PoppedResponse> {
        // Longest a single `wait_*` blocks before we re-poll. A backend whose
        // wait returns immediately (the emulator does) must not be mistaken for
        // a genuine timeout, so waits are sliced and the deadline is the sole
        // authority on giving up.
        let slice = Duration::from_millis(250);
        let deadline = Instant::now() + DEFAULT_TIMEOUT;

        // Push, retrying while the submission queue is full.
        loop {
            if self.transport.push_sq(sq_index, cmd, desc_flags)? {
                break;
            }
            if Instant::now() >= deadline {
                return Err(Error::Protocol(
                    "timed out waiting for submission-queue space".into(),
                ));
            }
            if !self.transport.wait_sq(remaining(deadline).min(slice))? {
                std::thread::sleep(Duration::from_millis(1));
            }
        }

        // Pop, skipping unrelated responses until the matching tag arrives.
        loop {
            if let Some(rsp) = self.transport.pop_cq()? {
                match proto::ResponseHeader::parse(&rsp.bytes) {
                    Some(hdr) if hdr.tag_id == expected_tag => return Ok(rsp),
                    // A response for another tag (e.g. an asynchronous event);
                    // drop it and keep waiting for ours.
                    _ => continue,
                }
            }
            if Instant::now() >= deadline {
                return Err(Error::Protocol(
                    "timed out waiting for command response".into(),
                ));
            }
            // A false return means no completion arrived in this slice, not a
            // fatal timeout; keep polling until the deadline.
            if !self.transport.wait_cq(remaining(deadline).min(slice))? {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }

    fn next_tag(&self) -> u16 {
        let t = self.tag.get();
        self.tag.set(t.wrapping_add(1));
        t
    }
}

/// Physical address for a DMA node at `offset` into a staging buffer whose base
/// physical address is `base`. A zero base means the backend resolves the
/// physical address itself, so it stays zero.
fn node_phys(base: u64, offset: usize) -> u64 {
    if base == 0 { 0 } else { base + offset as u64 }
}

fn align_up(value: u64, align: u64) -> u64 {
    debug_assert!(align.is_power_of_two());
    (value + align - 1) & !(align - 1)
}

fn remaining(deadline: Instant) -> Duration {
    deadline.saturating_duration_since(Instant::now())
}

fn parse_launch_timing(buf: &[u8]) -> LaunchTiming {
    let rd = |off: usize| -> u64 {
        buf.get(off..off + 8)
            .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
            .unwrap_or(0)
    };
    // Following the 8-byte response header: start_ts, execute_dur, wait_dur.
    LaunchTiming {
        start_ts: rd(8),
        execute_dur: rd(16),
        wait_dur: rd(24),
    }
}
