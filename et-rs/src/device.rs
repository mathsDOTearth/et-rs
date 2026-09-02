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
use crate::transport::{DeviceProperties, DramInfo, IoctlTransport, PoppedResponse, Transport};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
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

/// Options controlling a DMA transfer ([`Device::memcpy_h2d_opts`] /
/// [`Device::memcpy_d2h_opts`]).
///
/// ```
/// use et_soc1::DmaOptions;
/// // Route DMA to SQ 1 so it can run concurrently with a kernel on SQ 0.
/// let opts = DmaOptions::new().on_sq(1);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct DmaOptions {
    /// Submission queue to push DMA commands onto (default: 0).
    pub sq_index: u16,
}

impl Default for DmaOptions {
    fn default() -> Self {
        DmaOptions { sq_index: 0 }
    }
}

impl DmaOptions {
    /// Default DMA options: submission queue 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Route DMA commands to submission queue `idx`.
    ///
    /// Using a different queue from the kernel launch (which defaults to SQ 0)
    /// allows the firmware to process DMA and compute concurrently, enabling
    /// double-buffering patterns. Verify with [`Device::topology`] that the
    /// device has more than one queue before selecting `idx > 0`.
    pub fn on_sq(mut self, idx: u16) -> Self {
        self.sq_index = idx;
        self
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

    /// Clear the `BARRIER` flag so the firmware may start this kernel before
    /// all prior commands on the same SQ have completed.
    ///
    /// Use this when you have ensured ordering by other means -- for example,
    /// data was uploaded on a separate SQ (via [`DmaOptions::on_sq`]) and the
    /// kernel issues a `BARRIER` on its own queue to drain only the compute
    /// stream, not the DMA stream.
    ///
    /// The default constructor sets `barrier = true`; call `without_barrier()`
    /// last so it is not overridden by a later builder call.
    pub fn without_barrier(mut self) -> Self {
        self.barrier = false;
        self
    }

    /// Route this launch command to submission queue `idx`.
    ///
    /// The default is SQ 0. Routing a kernel launch to a dedicated compute
    /// queue while DMA goes to SQ 1 (via [`DmaOptions::on_sq`]) allows the
    /// firmware to schedule both streams concurrently.
    pub fn on_sq(mut self, idx: u16) -> Self {
        self.sq_index = idx;
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

/// Outcome of a successful [`Device::launch`] or [`Device::wait_launch`].
#[derive(Clone, Copy, Debug, Default)]
pub struct LaunchResult {
    /// Device timing counters for the launch.
    pub timing: LaunchTiming,
}

/// Handle to an in-flight kernel launch, returned by [`Device::launch_async`].
///
/// Pass to [`Device::wait_launch`] to block until the kernel completes and
/// retrieve its [`LaunchResult`]. The handle carries the CQ tag allocated for
/// the launch; dropping it without calling `wait_launch` leaves the response
/// in the device's completion stash until the next `wait_launch` or `launch`
/// clears it.
#[derive(Debug)]
pub struct PendingLaunch {
    tag: u16,
}

/// A connected ET-SoC-1 device.
pub struct Device<T: Transport = IoctlTransport> {
    transport: T,
    dram: DramInfo,
    /// Bump-allocation cursor within the user DRAM region.
    next: Cell<u64>,
    /// Reused device region for launch arguments, grown on demand so repeated
    /// launches do not leak a fresh region each time.
    args_scratch: Cell<Option<DeviceRegion>>,
    /// Monotonic command correlation tag.
    tag: Cell<u16>,
    /// Responses that arrived from the CQ for tags other than the one currently
    /// being waited for. Keyed by tag ID. Used by `collect_response` to park
    /// out-of-order responses so concurrent in-flight commands do not lose each
    /// other's completions (e.g. a kernel launch response arriving while a DMA
    /// command is being collected).
    stash: RefCell<HashMap<u16, PoppedResponse>>,
}

/// A saved position of the DRAM bump allocator, taken by [`Device::alloc_mark`]
/// and passed to [`Device::reset_to`] to reclaim everything allocated since.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AllocMark(u64);

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
            args_scratch: Cell::new(None),
            tag: Cell::new(0),
            stash: RefCell::new(HashMap::new()),
        })
    }

    /// The device's user DRAM region geometry and DMA limits.
    pub fn dram_info(&self) -> DramInfo {
        self.dram
    }

    /// The device's compute topology: the present compute-shire mask plus the
    /// architectural per-shire geometry. Query this to size work to the device
    /// instead of hard-coding shire masks and hart counts.
    pub fn topology(&self) -> Result<crate::Topology> {
        let cfg = self.transport.device_config()?;
        Ok(crate::Topology {
            shire_mask: cfg.shire_mask,
            harts_per_shire: crate::topology::HARTS_PER_SHIRE,
            harts_per_neighbourhood: crate::topology::HARTS_PER_NEIGHBOURHOOD,
            cache_line: cfg.cache_line,
        })
    }

    /// All device properties reported by `ETSOC1_IOCTL_GET_DEVICE_CONFIGURATION`.
    ///
    /// Includes cache sizes, DDR bandwidth, and `minion_boot_freq` (MHz), which
    /// can be used to convert a PMU cycle-count delta to wall time:
    ///
    /// ```text
    /// elapsed_us = cycles as f64 / (props.minion_boot_freq as f64);
    /// ```
    pub fn properties(&self) -> Result<DeviceProperties> {
        self.transport.device_properties()
    }

    /// Borrow the underlying transport.
    pub fn transport(&self) -> &T {
        &self.transport
    }

    /// Bytes still available in the DRAM bump allocator.
    pub fn dram_available(&self) -> u64 {
        (self.dram.base + self.dram.size).saturating_sub(self.next.get())
    }

    /// Allocate a region of device DRAM aligned to at least a cache line.
    ///
    /// This is a monotonic bump allocator: individual regions are not freed, but
    /// a span can be reclaimed as a group with [`Device::alloc_mark`] and
    /// [`Device::reset_to`]. Alignment is `max(dma_alignment, CACHE_LINE)`:
    /// the cache-line floor prevents two caller-allocated regions from sharing a
    /// line, which would cause silent false-sharing corruption on this
    /// software-coherent device even when neither region is accessed from the
    /// device concurrently.
    pub fn alloc(&self, size: u64) -> Result<DeviceRegion> {
        let align = self
            .dram
            .alignment()
            .max(et_abi::CACHE_LINE as u64);
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

    /// Record the current allocator position for a later [`Device::reset_to`].
    ///
    /// This is the arena/scratch pattern: mark a point, allocate freely, then
    /// reset to reclaim it all at once.
    pub fn alloc_mark(&self) -> AllocMark {
        AllocMark(self.next.get())
    }

    /// Reclaim every allocation made since `mark`, rewinding the bump allocator.
    ///
    /// Any [`DeviceRegion`] or [`DeviceBuffer`](crate::DeviceBuffer) obtained
    /// after `mark` must not be used afterwards: the DRAM it names may be handed
    /// out again. This cannot cause host-side undefined behaviour, but using a
    /// reclaimed region will read or overwrite unrelated device data. `mark` must
    /// come from this device; a mark ahead of the current position is ignored.
    pub fn reset_to(&self, mark: AllocMark) {
        if mark.0 < self.next.get() {
            self.next.set(mark.0);
        }
        // Drop the launch-args scratch if it now lies in the reclaimed span, so
        // the next launch re-allocates rather than reusing freed DRAM.
        if let Some(r) = self.args_scratch.get()
            && r.addr >= mark.0
        {
            self.args_scratch.set(None);
        }
    }

    /// Device region for a launch-argument payload of `len` bytes, reused across
    /// launches and grown on demand so repeated launches do not each leak a
    /// region. Reuse is safe because a launch runs to completion (the default
    /// barrier) before its scratch could be handed out again.
    fn args_region(&self, len: u64) -> Result<DeviceRegion> {
        if let Some(r) = self.args_scratch.get()
            && r.size >= len
        {
            return Ok(r);
        }
        let region = self.alloc(len)?;
        self.args_scratch.set(Some(region));
        Ok(region)
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

    /// Submit a kernel launch and return immediately without blocking.
    ///
    /// The kernel is pushed onto the device submission queue; it begins
    /// executing as soon as the firmware dequeues it. Pass the returned
    /// [`PendingLaunch`] to [`Device::wait_launch`] to block until completion.
    ///
    /// Between `launch_async` and `wait_launch` the caller may issue other
    /// device operations (DMA, a second launch, etc.); their responses are
    /// stashed and returned correctly when each is collected. This enables
    /// double-buffering patterns: while the kernel processes buffer A, the
    /// host can DMA-fill buffer B concurrently.
    ///
    /// If `opts.barrier` is `true` (the default), the firmware will drain all
    /// prior commands before starting this kernel. Set `barrier` to `false`
    /// only when the caller has ensured the prior operations (typically the
    /// DMA filling the kernel's input buffer) have completed.
    pub fn launch_async(
        &self,
        kernel: &LoadedKernel,
        opts: &LaunchOptions,
    ) -> Result<PendingLaunch> {
        let mut flags: u16 = 0;
        if opts.barrier {
            flags |= cmd_flags::BARRIER;
        }
        if opts.flush_l3 {
            flags |= cmd_flags::FLUSH_L3;
        }

        // Optional argument payload: trace configuration (40 B) then stack
        // configuration (8 B), each present only when the corresponding flag is set.
        let mut payload = Vec::new();
        if let Some(trace) = opts.trace {
            flags |= cmd_flags::COMPUTE_KERNEL_TRACE;
            payload.extend_from_slice(&trace.to_init_info().to_bytes());
        }
        if let Some(stack) = opts.stack {
            flags |= cmd_flags::USER_STACK_CFG;
            payload.extend_from_slice(&stack.to_bytes());
        }

        // Kernel arguments are delivered by pointer: the firmware passes the
        // device address of the args blob in `a0` at kernel entry. Stage the
        // bytes into device DRAM first, then encode the pointer in the command.
        // (Verified on device: `a0` carries the pointer; `ra` is 0 at entry
        // despite the SDK docs, and an embedded payload leaves neither populated.)
        let mut pointer_to_args: u64 = 0;
        if !opts.args.is_empty() {
            let region = self.args_region(opts.args.len() as u64)?;
            self.memcpy_h2d(&opts.args, region.addr)?;
            pointer_to_args = region.addr;
        }

        let tag = self.next_tag();
        let cmd = proto::build_kernel_launch(
            tag,
            flags,
            kernel.code_start_address,
            pointer_to_args,
            opts.exception_buffer,
            opts.shire_mask,
            &payload,
        );
        self.push_cmd(opts.sq_index, &cmd, 0)?;
        Ok(PendingLaunch { tag })
    }

    /// Block until a pending kernel launch completes and return its result.
    ///
    /// Any completion responses for other in-flight commands that arrive while
    /// waiting are stashed automatically and returned by their own `wait_launch`
    /// or subsequent `launch` call.
    pub fn wait_launch(&self, pending: PendingLaunch) -> Result<LaunchResult> {
        let rsp = self.collect_response(pending.tag)?;
        let status = proto::response_status(&rsp.bytes)
            .ok_or_else(|| Error::Protocol("kernel-launch response truncated".into()))?;
        if status != ops::DEV_OPS_API_KERNEL_LAUNCH_RESPONSE::DEV_OPS_API_KERNEL_LAUNCH_RESPONSE_KERNEL_COMPLETED {
            return Err(Error::KernelLaunch {
                status,
                status_name: proto::kernel_launch_status_name(status),
                detail: proto::parse_kernel_error_ptr(&rsp.bytes),
            });
        }
        Ok(LaunchResult {
            timing: parse_launch_timing(&rsp.bytes),
        })
    }

    /// Launch a loaded kernel and wait for its completion response.
    ///
    /// Equivalent to [`Device::launch_async`] followed immediately by
    /// [`Device::wait_launch`]. Use `launch_async` + `wait_launch` directly
    /// when overlapping computation with DMA or issuing multiple concurrent
    /// launches.
    pub fn launch(&self, kernel: &LoadedKernel, opts: &LaunchOptions) -> Result<LaunchResult> {
        let pending = self.launch_async(kernel, opts)?;
        self.wait_launch(pending)
    }

    /// Launch `kernel` across `shire_mask` (SPMD) with typed arguments.
    ///
    /// This bundles the argument staging that [`Device::launch`] otherwise does by
    /// hand: `args` is the shared [`et_abi::DeviceArgs`] struct the kernel reads,
    /// serialised and delivered by pointer (the firmware passes its address in
    /// `a0`). Equivalent to `launch(kernel, &LaunchOptions::new(shire_mask)
    /// .with_args(args.as_bytes().to_vec()))`.
    pub fn launch_spmd<A: et_abi::DeviceArgs>(
        &self,
        kernel: &LoadedKernel,
        shire_mask: u64,
        args: &A,
    ) -> Result<LaunchResult> {
        let opts = LaunchOptions::new(shire_mask).with_args(args.as_bytes().to_vec());
        self.launch(kernel, &opts)
    }

    /// As [`Device::launch_spmd`], additionally capturing a U-mode trace.
    pub fn launch_spmd_traced<A: et_abi::DeviceArgs>(
        &self,
        kernel: &LoadedKernel,
        shire_mask: u64,
        args: &A,
        trace: TraceConfig,
    ) -> Result<LaunchResult> {
        let opts = LaunchOptions::new(shire_mask)
            .with_args(args.as_bytes().to_vec())
            .with_trace(trace);
        self.launch(kernel, &opts)
    }

    /// Copy `dst.len()` bytes from device address `src` into host memory via a
    /// DMA read-list command, splitting the transfer to honour the device's DMA
    /// element-size and element-count limits.
    ///
    /// The transfer is staged through a transport-provided DMA host buffer (see
    /// [`crate::transport::DmaHostBuffer`]) and copied out afterwards, so it works
    /// whether the backend pins arbitrary host memory or requires registered DMA
    /// memory.
    ///
    /// Intermediate DMA batches (when the transfer spans more than
    /// `dma_max_elem_count` nodes) are issued without the `BARRIER` flag so the
    /// firmware's DMA engine can pipeline them; only the final batch uses
    /// `BARRIER` to ensure completion before the function returns.
    ///
    /// Uses the default [`DmaOptions`] (SQ 0). Use [`Device::memcpy_d2h_opts`]
    /// to select a different submission queue.
    pub fn memcpy_d2h(&self, src: u64, dst: &mut [u8]) -> Result<()> {
        self.memcpy_d2h_opts(src, dst, &DmaOptions::default())
    }

    /// Copy `dst.len()` bytes from device address `src` into host memory,
    /// routing DMA commands according to `opts`.
    ///
    /// Equivalent to [`Device::memcpy_d2h`] but with explicit [`DmaOptions`].
    /// Use `opts.on_sq(1)` to send DMA to a separate submission queue, enabling
    /// concurrent DMA and compute when paired with [`Device::launch_async`].
    pub fn memcpy_d2h_opts(&self, src: u64, dst: &mut [u8], opts: &DmaOptions) -> Result<()> {
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
                // Barrier only on the final batch: earlier batches may overlap
                // with subsequent firmware DMA scheduling for better throughput.
                let is_last = offset >= total;
                let flags = if is_last { cmd_flags::BARRIER } else { 0 };
                self.dma_read_command(&nodes, flags, opts.sq_index)?;
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
    ///
    /// Intermediate batches are issued without `BARRIER` (see [`memcpy_d2h`]);
    /// only the final batch uses `BARRIER` to guarantee completion on return.
    ///
    /// Uses the default [`DmaOptions`] (SQ 0). Use [`Device::memcpy_h2d_opts`]
    /// to select a different submission queue.
    pub fn memcpy_h2d(&self, src: &[u8], dst: u64) -> Result<()> {
        self.memcpy_h2d_opts(src, dst, &DmaOptions::default())
    }

    /// Copy `src.len()` bytes from host memory to device address `dst`,
    /// routing DMA commands according to `opts`.
    ///
    /// Equivalent to [`Device::memcpy_h2d`] but with explicit [`DmaOptions`].
    /// Use `opts.on_sq(1)` to send DMA to a separate submission queue, enabling
    /// concurrent DMA and compute when paired with [`Device::launch_async`].
    pub fn memcpy_h2d_opts(&self, src: &[u8], dst: u64, opts: &DmaOptions) -> Result<()> {
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
                let is_last = offset >= total;
                let flags = if is_last { cmd_flags::BARRIER } else { 0 };
                self.dma_write_command(&nodes, flags, opts.sq_index)?;
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

    fn dma_read_command(&self, nodes: &[proto::DmaReadNode], flags: u16, sq_index: u16) -> Result<()> {
        let tag = self.next_tag();
        let cmd = proto::build_dma_readlist(tag, flags, nodes);
        let rsp = self.submit(sq_index, &cmd, desc_flags::DMA, tag)?;
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

    fn dma_write_command(&self, nodes: &[proto::DmaWriteNode], flags: u16, sq_index: u16) -> Result<()> {
        let tag = self.next_tag();
        let cmd = proto::build_dma_writelist(tag, flags, nodes);
        let rsp = self.submit(sq_index, &cmd, desc_flags::DMA, tag)?;
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

    /// Push a command onto the submission queue, retrying until space is available.
    fn push_cmd(
        &self,
        sq_index: u16,
        cmd: &[u8],
        desc_flags: u8,
    ) -> Result<()> {
        // Longest a single `wait_sq` blocks before re-polling. A backend whose
        // wait returns immediately (the emulator does) must not be mistaken for
        // a genuine timeout; the deadline is the sole authority on giving up.
        let slice = Duration::from_millis(250);
        let deadline = Instant::now() + DEFAULT_TIMEOUT;
        loop {
            if self.transport.push_sq(sq_index, cmd, desc_flags)? {
                return Ok(());
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
    }

    /// Block until the CQ response bearing `expected_tag` arrives.
    ///
    /// Responses for other in-flight tags are placed in `self.stash` so that
    /// concurrent `launch_async` / `wait_launch` sequences do not discard each
    /// other's completions.
    fn collect_response(&self, expected_tag: u16) -> Result<PoppedResponse> {
        let slice = Duration::from_millis(250);
        let deadline = Instant::now() + DEFAULT_TIMEOUT;

        // Check the stash before polling the CQ: if a prior `collect_response`
        // parked this tag, return it immediately without touching the hardware.
        if let Some(rsp) = self.stash.borrow_mut().remove(&expected_tag) {
            return Ok(rsp);
        }

        loop {
            if let Some(rsp) = self.transport.pop_cq()? {
                match proto::ResponseHeader::parse(&rsp.bytes) {
                    Some(hdr) if hdr.tag_id == expected_tag => return Ok(rsp),
                    Some(hdr) => {
                        // Response for a different in-flight command; park it.
                        self.stash.borrow_mut().insert(hdr.tag_id, rsp);
                    }
                    None => {} // Malformed response; discard silently.
                }
                continue;
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

    /// Push a command and block for the response bearing `expected_tag`.
    ///
    /// A thin wrapper around [`push_cmd`] + [`collect_response`]; used for
    /// operations that are inherently synchronous (DMA commands, where the
    /// caller needs the result before continuing).
    fn submit(
        &self,
        sq_index: u16,
        cmd: &[u8],
        desc_flags: u8,
        expected_tag: u16,
    ) -> Result<PoppedResponse> {
        self.push_cmd(sq_index, cmd, desc_flags)?;
        self.collect_response(expected_tag)
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
