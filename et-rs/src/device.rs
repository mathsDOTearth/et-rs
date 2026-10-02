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
use crate::transport::{
    DeviceProperties, DmaHostBuffer, DramInfo, IoctlTransport, PoppedResponse, Transport,
};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::thread;
use std::time::{Duration, Instant};

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
    ///
    /// The launch command carries the buffer size as a `u32`; a region larger
    /// than 4 GiB is clamped to the largest cache-line multiple that fits.
    pub fn full(buffer: DeviceRegion, shire_mask: u64) -> Self {
        const MAX_TRACE_BYTES: u64 = (u32::MAX as u64) & !(et_abi::CACHE_LINE as u64 - 1);
        TraceConfig {
            buffer: buffer.addr,
            buffer_size: buffer.size.min(MAX_TRACE_BYTES) as u32,
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
/// use std::time::Duration;
/// // Route DMA to SQ 1, with a 60 s timeout for a large transfer.
/// let opts = DmaOptions::new().on_sq(1).with_timeout(Duration::from_secs(60));
/// ```
///
/// `#[non_exhaustive]`: additional fields may be added in future patch releases.
#[derive(Clone, Copy, Debug, Default)]
#[non_exhaustive]
pub struct DmaOptions {
    /// Submission queue to push DMA commands onto (default: 0).
    pub sq_index: u16,
    /// Maximum duration to wait for DMA completion.
    ///
    /// `None` (the default) means wait indefinitely. Use
    /// [`DmaOptions::with_timeout`] to set an upper bound.
    timeout: Option<Duration>,
}

impl DmaOptions {
    /// Default DMA options: submission queue 0, unlimited completion wait.
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

    /// Set the maximum duration to wait for a DMA completion response.
    ///
    /// By default, DMA waits are unlimited: the call returns as soon as the
    /// firmware acknowledges the transfer. Use this method to impose a bound for
    /// defensive error detection on DMA paths that should be fast.
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = Some(t);
        self
    }
}

/// Options controlling a single kernel launch.
///
/// `#[non_exhaustive]`: additional fields may be added in future patch releases.
/// Construct via [`LaunchOptions::new`] and the provided builder methods.
#[derive(Clone, Debug)]
#[non_exhaustive]
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
    /// Maximum duration to wait for the kernel completion response.
    /// `None` inherits the device-level default from
    /// [`Transport::default_launch_timeout`], resolved at launch time.
    timeout: Option<Duration>,
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
            timeout: None,
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

    /// Set the maximum duration to wait for the kernel completion response.
    ///
    /// The default is 10 s, which is adequate for short test kernels but
    /// insufficient for production workloads (large matrix multiplications,
    /// multi-frame radiosity solves, etc.). Pass a duration at least as large
    /// as the worst-case kernel execution time; erring on the side of a longer
    /// timeout is safe -- the call returns as soon as the response arrives.
    ///
    /// ```
    /// use et_soc1::LaunchOptions;
    /// use std::time::Duration;
    ///
    /// let opts = LaunchOptions::new(0x1)
    ///     .with_timeout(Duration::from_secs(300)); // 5 min for a long kernel
    /// ```
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = Some(t);
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
/// in the device's completion stash, and its command tag reserved, for the
/// lifetime of the [`Device`].
#[derive(Debug)]
pub struct PendingLaunch {
    tag: u16,
    /// Tags of the DMA commands staging the launch arguments, pushed ahead of
    /// the launch and collected by [`Device::wait_launch`]. Empty when the
    /// launch has no arguments or when they were awaited before the launch.
    args_dma: Vec<u16>,
    /// Timeout for [`Device::wait_launch`], copied from [`LaunchOptions::timeout`].
    timeout: Duration,
}

/// A host buffer that the device reads and writes directly by DMA, created by
/// [`Device::alloc_pinned`].
///
/// [`Device::memcpy_h2d`] and [`Device::memcpy_d2h`] copy between the caller's
/// memory and an internal staging buffer, which costs about as much as the DMA
/// itself for large transfers. A `PinnedBuffer` is a staging buffer held by
/// the caller: data is produced or consumed in place, and
/// [`Device::memcpy_h2d_pinned`] and [`Device::memcpy_d2h_pinned`] issue the
/// DMA with no host copy.
///
/// The buffer borrows the device that created it, since its mapping belongs
/// to that device's transport. If a transfer fails while the device may still
/// be accessing the buffer (a timeout or transport error), the mapping is
/// deliberately leaked and the buffer becomes empty, so the device can never
/// access memory the host has since reused.
pub struct PinnedBuffer<'d> {
    host: Option<Box<dyn DmaHostBuffer>>,
    len: usize,
    /// Address of the creating [`Device`], checked on every transfer.
    device: *const (),
    _device: std::marker::PhantomData<&'d ()>,
}

impl PinnedBuffer<'_> {
    /// Length of the buffer in bytes; zero after a failed in-flight transfer.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The buffer contents.
    pub fn as_slice(&self) -> &[u8] {
        match &self.host {
            Some(host) => &host.as_slice()[..self.len],
            None => &[],
        }
    }

    /// The buffer contents, for writing.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        match &mut self.host {
            Some(host) => &mut host.as_mut_slice()[..self.len],
            None => &mut [],
        }
    }
}

impl std::fmt::Debug for PinnedBuffer<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedBuffer")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

/// Default capacity, in bytes, of the persistent DMA staging buffer; see
/// [`Device::set_staging_capacity`].
pub const DEFAULT_STAGING_CAPACITY: usize = 16 << 20;

/// A connected ET-SoC-1 device.
pub struct Device<T: Transport = IoctlTransport> {
    /// Host DMA staging buffers retained between transfers, so that each
    /// `memcpy_h2d`/`memcpy_d2h` does not map and unmap fresh driver buffers:
    /// one for transfers of up to half the staging capacity, a second for
    /// pipelining larger ones. Empty until the first transfer, or after an
    /// in-flight failure leaked them. Declared before `transport` so that they
    /// are released first on drop.
    staging: RefCell<Vec<Box<dyn DmaHostBuffer>>>,
    /// Upper bound on the combined size of `staging`. Always a non-zero
    /// multiple of twice the DMA alignment, so each half is aligned.
    staging_capacity: Cell<usize>,
    /// Device regions used to stage launch arguments, each tagged with the
    /// command that last used it. A slot is reused only once that command's
    /// completion has been collected, so neither a running kernel's arguments
    /// nor an in-flight argument DMA is ever overwritten by a later launch.
    /// Declared before `transport` because each slot holds a host DMA buffer.
    args_slots: RefCell<Vec<ArgsSlot>>,
    transport: T,
    dram: DramInfo,
    /// Bump-allocation cursor within the user DRAM region.
    next: Cell<u64>,
    /// End of the DRAM occupied by loaded kernel images. [`Device::reset_to`]
    /// never rewinds below this point.
    kernel_end: Cell<u64>,
    /// Next candidate command correlation tag.
    tag: Cell<u16>,
    /// Tags of commands pushed but whose responses have not yet been consumed
    /// (including abandoned ones). A tag is not reissued while it is here, so a
    /// late response can never be mistaken for the reply to a newer command
    /// after the 16-bit tag space wraps.
    outstanding: RefCell<HashSet<u16>>,
    /// Outstanding tags whose waiter gave up (timeout or transport error). The
    /// late response, if it ever arrives, is discarded and the tag released.
    abandoned: RefCell<HashSet<u16>>,
    /// Largest DMA list (in nodes) a single command may carry, derived lazily
    /// from the DMA element-count limit, the `u16` command-size field and the
    /// submission-queue message size.
    max_list_nodes: Cell<Option<usize>>,
    /// Responses that arrived from the CQ for tags other than the one currently
    /// being waited for. Keyed by tag ID. Used by `collect_response` to park
    /// out-of-order responses so concurrent in-flight commands do not lose each
    /// other's completions (e.g. a kernel launch response arriving while a DMA
    /// command is being collected).
    stash: RefCell<HashMap<u16, PoppedResponse>>,
    /// Default timeout for kernel completion waits (not DMA, which is unlimited
    /// unless overridden via [`DmaOptions::with_timeout`]).
    /// Initialised from [`Transport::default_launch_timeout`]; may be overridden
    /// at runtime via [`Device::set_default_launch_timeout`].
    default_timeout: Cell<Duration>,
}

/// A launch-argument staging region, the host DMA buffer from which arguments
/// are copied into it, and the tag of the command that last used it: the
/// argument DMA until its launch has been pushed, the launch thereafter.
struct ArgsSlot {
    region: DeviceRegion,
    /// Host buffer of at least `region.size` bytes, mapped on first use.
    host: Option<Box<dyn DmaHostBuffer>>,
    owner: Option<u16>,
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

    /// Trigger a full ETSOC device reset and return a freshly opened device.
    ///
    /// Perform a full ETSOC device reset and return a fresh handle.
    ///
    /// Consumes `self` so the ops device node is closed before the firmware
    /// completes the reset (the driver requires this: it will not finish while
    /// any file descriptor to the ops node remains open).
    ///
    /// The sequence is:
    /// 1. Derive the management device path from the ops path
    ///    (`/dev/etN_ops` -> `/dev/etN_mgmt`).
    /// 2. Drop `self`, closing the ops fd.
    /// 3. Open `/dev/etN_mgmt` and submit a `device_mgmt_etsoc_reset_cmd_t`
    ///    via `PUSH_SQ` with `CMD_DESC_FLAG_ETSOC_RESET`. The firmware
    ///    initiates the reset on receipt; no response arrives.
    /// 4. Poll `/dev/etN_ops` every 250 ms until the device is accessible
    ///    again, up to a 30 s deadline.
    /// 5. Re-open and return a new `Device<IoctlTransport>`.
    ///
    /// Note: `CMD_DESC_FLAG_ETSOC_RESET` is rejected (EINVAL) on the ops
    /// node; it is only valid on the Service Processor management node.
    ///
    /// Returns [`Error::Protocol`] if the transport was constructed via
    /// [`IoctlTransport::from_owned_fd`] (path is unknown in that case).
    /// Returns [`Error::Timeout`] if the device does not come back within 30 s.
    ///
    /// Use [`Device::reset_shires`] instead when only the compute minions need
    /// recovery and the firmware is still responsive.
    pub fn reset_device(self) -> Result<Self> {
        // Capture the ops path before consuming self.
        let path = self
            .transport
            .device_path()
            .ok_or_else(|| {
                Error::Protocol(
                    "reset_device requires a path-based transport; \
                     use Device::open or IoctlTransport::open_path"
                        .into(),
                )
            })?
            .to_path_buf();

        // Derive the management node path: /dev/et0_ops -> /dev/et0_mgmt.
        // The driver creates one mgmt node per ops node under the same /dev.
        let mgmt_path = crate::transport::mgmt_path_for(&path).ok_or_else(|| {
            Error::Protocol(format!(
                "cannot derive the management node from {}",
                path.display()
            ))
        })?;

        // Build the ETSOC reset command (device_mgmt_etsoc_reset_cmd_t, 16 B).
        let tag = self.next_tag()?;
        let cmd = proto::build_etsoc_reset(tag);

        // Close the ops fd. The driver will not complete the reset while any
        // ops fd remains open; this drop satisfies that requirement.
        drop(self);

        // Submit the reset via the management node with ETSOC_RESET flag.
        // The management node accepts this flag; the ops node returns EINVAL.
        IoctlTransport::push_one_cmd(&mgmt_path, &cmd, desc_flags::ETSOC_RESET)?;

        // Poll until the ops node opens and answers the device queries (reset
        // complete) or the 30 s deadline elapses. The node may become openable
        // before the firmware is ready to serve queries, so a failure of
        // `with_transport` is retried in the same way as a failed open.
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            thread::sleep(Duration::from_millis(250));
            match IoctlTransport::open_path(&path).and_then(Device::with_transport) {
                Ok(device) => return Ok(device),
                Err(_) if Instant::now() < deadline => continue,
                Err(_) => {
                    return Err(Error::Timeout {
                        operation: "device reset",
                        limit: Duration::from_secs(30),
                    });
                }
            }
        }
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
        let default_timeout = transport.default_launch_timeout();
        let staging_capacity = align_staging_capacity(DEFAULT_STAGING_CAPACITY, &dram);
        Ok(Device {
            staging: RefCell::new(Vec::new()),
            staging_capacity: Cell::new(staging_capacity),
            transport,
            dram,
            next: Cell::new(dram.base),
            args_slots: RefCell::new(Vec::new()),
            kernel_end: Cell::new(dram.base),
            tag: Cell::new(0),
            outstanding: RefCell::new(HashSet::new()),
            abandoned: RefCell::new(HashSet::new()),
            max_list_nodes: Cell::new(None),
            stash: RefCell::new(HashMap::new()),
            default_timeout: Cell::new(default_timeout),
        })
    }

    /// Set the combined capacity of the persistent host DMA staging buffers, in
    /// bytes, and release the current buffers.
    ///
    /// Every `memcpy_h2d`/`memcpy_d2h` passes through host buffers that the
    /// transport can DMA to and from. They are allocated on first use, grown in
    /// powers of two up to half this capacity each, and retained between
    /// transfers. A transfer of up to half the capacity uses one buffer; a
    /// larger one is split into half-capacity chunks and pipelined through two
    /// buffers, so that the host copy of one chunk overlaps the DMA of the
    /// other. A larger capacity costs fewer DMA commands per transfer at the
    /// price of more pinned host memory (driver CMA memory on the ioctl
    /// transport) held for the life of the device. The default is
    /// [`DEFAULT_STAGING_CAPACITY`].
    ///
    /// `bytes` is rounded down to a multiple of twice the device's DMA
    /// alignment, so that every chunk starts at an aligned device address, and
    /// is raised to that quantum if smaller.
    pub fn set_staging_capacity(&self, bytes: usize) {
        self.staging_capacity
            .set(align_staging_capacity(bytes, &self.dram));
        self.staging.borrow_mut().clear();
    }

    /// Current combined capacity of the persistent host DMA staging buffers,
    /// in bytes.
    pub fn staging_capacity(&self) -> usize {
        self.staging_capacity.get()
    }

    /// Override the default kernel completion timeout for this device.
    ///
    /// The initial value comes from [`crate::transport::Transport::default_launch_timeout`]:
    /// 10 s for [`IoctlTransport`] (real hardware) and 1 h for the FFI emulator.
    /// Per-launch timeouts set via [`LaunchOptions::with_timeout`] take
    /// precedence over this device-level default.
    ///
    /// DMA timeouts are not affected by this setting; use
    /// [`DmaOptions::with_timeout`] to bound an individual DMA transfer.
    ///
    /// Use this when all (or most) kernels run longer than the transport default
    /// and adding `.with_timeout` to every `LaunchOptions` would be repetitive.
    pub fn set_default_launch_timeout(&self, timeout: Duration) {
        self.default_timeout.set(timeout);
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
    /// if let Some(freq_mhz) = props.minion_boot_freq {
    ///     elapsed_us = cycles as f64 / (freq_mhz as f64);
    /// }
    /// ```
    ///
    /// `minion_boot_freq` is `None` when the transport cannot provide a clock
    /// value (e.g. the default or emulator transport).
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
        let align = self.dram.alignment().max(et_abi::CACHE_LINE as u64);
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
    ///
    /// The allocator never rewinds below the end of the loaded kernel images:
    /// a mark taken before [`Device::load_kernel`] reclaims only the
    /// allocations made after the kernel, so kernel code is never handed out.
    pub fn reset_to(&self, mark: AllocMark) {
        let target = mark.0.max(self.kernel_end.get());
        if target < self.next.get() {
            self.next.set(target);
        }
        // Forget argument slots in the reclaimed span, so the next launch
        // re-allocates rather than reusing freed DRAM. The host buffer of a
        // slot whose command is still outstanding is leaked rather than
        // unmapped, since an argument DMA may still be reading it.
        let outstanding = self.outstanding.borrow();
        self.args_slots.borrow_mut().retain_mut(|slot| {
            if slot.region.addr < target {
                return true;
            }
            if slot.owner.is_some_and(|tag| outstanding.contains(&tag))
                && let Some(host) = slot.host.take()
            {
                std::mem::forget(host);
            }
            false
        });
    }

    /// Index into `args_slots` of a staging region for a launch-argument payload
    /// of `len` bytes. A slot is reusable only when the command that last used
    /// it is no longer outstanding: its completion has been collected, so
    /// neither the kernel nor the argument DMA can still be accessing it. The
    /// smallest reusable slot that fits is chosen; otherwise a new region is
    /// allocated. The number of slots is therefore bounded by the number of
    /// launches in flight.
    fn args_slot(&self, len: u64) -> Result<usize> {
        {
            let outstanding = self.outstanding.borrow();
            let slots = self.args_slots.borrow();
            let reusable = slots
                .iter()
                .enumerate()
                .filter(|(_, slot)| {
                    slot.region.size >= len
                        && slot.owner.is_none_or(|tag| !outstanding.contains(&tag))
                })
                .min_by_key(|(_, slot)| slot.region.size)
                .map(|(i, _)| i);
            if let Some(i) = reusable {
                return Ok(i);
            }
        }
        let region = self.alloc(len)?;
        let mut slots = self.args_slots.borrow_mut();
        slots.push(ArgsSlot {
            region,
            host: None,
            owner: None,
        });
        Ok(slots.len() - 1)
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
    /// Call this before any [`Device::alloc`] or [`Device::alloc_padded`] calls.
    /// The kernel ELF is linked at the DRAM base address and this function DMA-
    /// writes each `PT_LOAD` segment to its `p_vaddr`.
    ///
    /// Every segment is validated before any DMA is issued: a segment outside
    /// the DRAM region, or one overlapping memory allocated since the last
    /// kernel load, is rejected with [`Error::Limit`] and nothing is written.
    /// A later kernel may replace an earlier one at the same address, provided
    /// it does not extend into allocations made after the earlier load.
    pub fn load_kernel(&self, elf_image: &[u8]) -> Result<LoadedKernel> {
        /// Size of the zero block reused to clear `.bss` tails, bounding the
        /// host allocation independently of the segment size.
        const ZERO_CHUNK: usize = 1 << 20;

        let image = elf::parse(elf_image)?;
        let region_end = self.dram.base + self.dram.size;
        let live_start = self.kernel_end.get();
        let live_end = self.next.get();
        let mut occupied_end = live_start;

        // `elf::parse` guarantees `vaddr + mem_size` does not overflow and that
        // each file range lies within the image.
        for seg in &image.segments {
            let seg_end = seg.vaddr + seg.mem_size;
            if seg.vaddr < self.dram.base || seg_end > region_end {
                return Err(Error::Limit(format!(
                    "kernel segment [{:#x}, {:#x}) lies outside the DRAM region [{:#x}, {:#x})",
                    seg.vaddr, seg_end, self.dram.base, region_end
                )));
            }
            if seg.mem_size > 0 && seg.vaddr < live_end && live_start < seg_end {
                return Err(Error::Limit(format!(
                    "kernel segment [{:#x}, {:#x}) overlaps live allocations [{:#x}, {:#x}); \
                     load kernels before allocating, or reset_to an earlier mark first",
                    seg.vaddr, seg_end, live_start, live_end
                )));
            }
            occupied_end = occupied_end.max(seg_end);
        }

        for seg in &image.segments {
            if seg.file_size > 0 {
                let start = seg.file_offset as usize;
                let end = start + seg.file_size as usize;
                self.memcpy_h2d(&elf_image[start..end], seg.vaddr)?;
            }
            // Zero-initialise the `.bss` tail present in memory but not in the file.
            if seg.mem_size > seg.file_size {
                let tail = seg.mem_size - seg.file_size;
                let zeros = vec![0u8; (tail as usize).min(ZERO_CHUNK)];
                let mut done = 0u64;
                while done < tail {
                    let len = (tail - done).min(zeros.len() as u64) as usize;
                    self.memcpy_h2d(&zeros[..len], seg.vaddr + seg.file_size + done)?;
                    done += len as u64;
                }
            }
        }

        // Reserve the DRAM the kernel occupies against future allocations.
        let align = self.dram.alignment().max(1);
        let kernel_end = align_up(occupied_end, align).min(region_end);
        self.kernel_end.set(kernel_end);
        self.next.set(self.next.get().max(kernel_end));

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
        // device address of the args blob in `a0` at kernel entry. The bytes
        // are DMA-written into device DRAM on the launch's submission queue,
        // and the pointer encoded in the command. (Verified on device: `a0`
        // carries the pointer; `ra` is 0 at entry despite the SDK docs, and an
        // embedded payload leaves neither populated.)
        //
        // With `barrier` set the launch cannot start before the argument DMA
        // completes, so the DMA is pushed without being awaited and its
        // completion is collected by `wait_launch`. Without `barrier` the
        // launch could overtake it, so it is awaited here.
        let mut pointer_to_args: u64 = 0;
        let mut args_slot = None;
        let mut args_dma = Vec::new();
        if !opts.args.is_empty() {
            let slot = self.args_slot(opts.args.len() as u64)?;
            let staged = self
                .push_args_dma(slot, &opts.args, opts.sq_index, &mut args_dma)
                .and_then(|addr| {
                    if !opts.barrier {
                        let count = args_dma.len();
                        self.await_dma(&mut args_dma, count, opts.timeout, "dma-writelist")?;
                    }
                    Ok(addr)
                });
            match staged {
                Ok(addr) => pointer_to_args = addr,
                Err(e) => {
                    self.abandon_all(&mut args_dma);
                    return Err(e);
                }
            }
            args_slot = Some(slot);
        }

        let tag = match self.next_tag() {
            Ok(tag) => tag,
            Err(e) => {
                self.abandon_all(&mut args_dma);
                return Err(e);
            }
        };
        let cmd = proto::build_kernel_launch(
            tag,
            flags,
            kernel.code_start_address,
            pointer_to_args,
            opts.exception_buffer,
            opts.shire_mask,
            &payload,
        );
        if let Err(e) = self.push_cmd(opts.sq_index, &cmd, 0, tag) {
            self.abandon_all(&mut args_dma);
            return Err(e);
        }
        // The slot now belongs to this launch until its completion is collected.
        // The launch carries `BARRIER` whenever `args_dma` is non-empty, so its
        // completion implies that of the argument DMA.
        if let Some(slot) = args_slot {
            self.args_slots.borrow_mut()[slot].owner = Some(tag);
        }
        Ok(PendingLaunch {
            tag,
            args_dma,
            timeout: opts.timeout.unwrap_or_else(|| self.default_timeout.get()),
        })
    }

    /// Block until a pending kernel launch completes and return its result.
    ///
    /// Any completion responses for other in-flight commands that arrive while
    /// waiting are stashed automatically and returned by their own `wait_launch`
    /// or subsequent `launch` call.
    ///
    /// The timeout applied is the one set via [`LaunchOptions::with_timeout`]
    /// when the launch was submitted (default 10 s).
    pub fn wait_launch(&self, mut pending: PendingLaunch) -> Result<LaunchResult> {
        // The argument DMA precedes the launch, so its completion is collected
        // first. On failure the launch is abandoned: its tag stays reserved, and
        // its argument slot unavailable, until the late response arrives.
        let count = pending.args_dma.len();
        let staged = self.await_dma(
            &mut pending.args_dma,
            count,
            Some(pending.timeout),
            "dma-writelist",
        );
        if let Err(e) = staged {
            self.abandon_all(&mut pending.args_dma);
            self.abandon(pending.tag);
            return Err(e);
        }
        let rsp = self.collect_response(pending.tag, Some(pending.timeout), "kernel completion")?;
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

    /// As [`Device::launch_spmd`] with explicit launch options.
    ///
    /// The `shire_mask` is taken from `opts`; `args` is serialised and appended
    /// via [`LaunchOptions::with_args`], overwriting any args already in `opts`.
    /// Use this to set a per-launch timeout or a non-default submission queue:
    ///
    /// ```
    /// use et_soc1::{LaunchOptions, LoadedKernel};
    /// use std::time::Duration;
    ///
    /// # fn example(device: &et_soc1::Device, kernel: &LoadedKernel, args: &impl et_abi::DeviceArgs) -> et_soc1::Result<et_soc1::LaunchResult> {
    /// device.launch_spmd_opts(
    ///     kernel,
    ///     args,
    ///     LaunchOptions::new(0xFFFF_FFFF).with_timeout(Duration::from_secs(120)),
    /// )
    /// # }
    /// ```
    pub fn launch_spmd_opts<A: et_abi::DeviceArgs>(
        &self,
        kernel: &LoadedKernel,
        args: &A,
        opts: LaunchOptions,
    ) -> Result<LaunchResult> {
        self.launch(kernel, &opts.with_args(args.as_bytes().to_vec()))
    }

    /// As [`Device::launch_spmd_traced`] with explicit launch options.
    ///
    /// The `shire_mask` is taken from `opts`; `args` is appended via
    /// [`LaunchOptions::with_args`] and `trace` via [`LaunchOptions::with_trace`],
    /// overwriting any previously set values in `opts`.
    pub fn launch_spmd_traced_opts<A: et_abi::DeviceArgs>(
        &self,
        kernel: &LoadedKernel,
        args: &A,
        trace: TraceConfig,
        opts: LaunchOptions,
    ) -> Result<LaunchResult> {
        let opts = opts.with_args(args.as_bytes().to_vec()).with_trace(trace);
        self.launch(kernel, &opts)
    }

    /// Copy `dst.len()` bytes from device address `src` into host memory via a
    /// DMA read-list command, splitting the transfer to honour the device's DMA
    /// element-size and element-count limits.
    ///
    /// The transfer is staged through the device's persistent DMA host buffers
    /// (see [`Device::set_staging_capacity`] and
    /// [`crate::transport::DmaHostBuffer`]) and copied out afterwards, so it
    /// works whether the backend pins arbitrary host memory or requires
    /// registered DMA memory.
    ///
    /// A transfer larger than half the staging capacity is pipelined through
    /// two half-capacity buffers: the DMA of the next chunk is in flight while
    /// the previous chunk is copied out of the other buffer. A chunk needing
    /// more list nodes than one command can carry is split into several
    /// commands. Every command carries `BARRIER`, so none starts before
    /// previously submitted work on its queue has completed.
    ///
    /// If a command times out or the transport fails, or fails while another
    /// command is still queued, the device may still be writing into the
    /// staging buffers; they are then leaked rather than returned to the
    /// driver, so the DMA cannot land in reallocated host memory, and fresh
    /// buffers are allocated by the next transfer.
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
        let mut buffers = self.take_staging(total)?;
        let mut pending = Vec::new();
        let result = self.d2h_pipeline(&buffers, src, dst, opts, &mut pending);
        self.restore_staging(&mut buffers, &mut pending, &result);
        result
    }

    /// Map a host buffer of `len` bytes that the device can access directly by
    /// DMA; see [`PinnedBuffer`]. The contents are initially unspecified (zero
    /// on the current transports).
    ///
    /// On the kernel-driver transport each buffer is a fresh allocation from
    /// the driver's contiguous DMA region, so buffers should be allocated once
    /// and reused rather than created per transfer.
    pub fn alloc_pinned(&self, len: usize) -> Result<PinnedBuffer<'_>> {
        Ok(PinnedBuffer {
            host: Some(self.transport.dma_host_buffer(len)?),
            len,
            device: self as *const Self as *const (),
            _device: std::marker::PhantomData,
        })
    }

    /// Write the whole of `src` to device address `dst` by DMA directly from
    /// the pinned buffer, with no host copy.
    pub fn memcpy_h2d_pinned(&self, src: &mut PinnedBuffer<'_>, dst: u64) -> Result<()> {
        self.memcpy_h2d_pinned_opts(src, dst, &DmaOptions::default())
    }

    /// [`Device::memcpy_h2d_pinned`] with explicit [`DmaOptions`].
    ///
    /// `src` is borrowed mutably, although only read, so that it cannot be
    /// modified during the transfer and can be emptied if the transfer fails
    /// with DMA possibly still in flight.
    pub fn memcpy_h2d_pinned_opts(
        &self,
        src: &mut PinnedBuffer<'_>,
        dst: u64,
        opts: &DmaOptions,
    ) -> Result<()> {
        self.pinned_transfer(src, |device, host, len, pending| {
            device.push_dma_write_staged(host, len, dst, opts, pending)?;
            let count = pending.len();
            device.await_dma(pending, count, opts.timeout, "dma-writelist")
        })
    }

    /// Fill the whole of `dst` from device address `src` by DMA directly into
    /// the pinned buffer, with no host copy.
    pub fn memcpy_d2h_pinned(&self, src: u64, dst: &mut PinnedBuffer<'_>) -> Result<()> {
        self.memcpy_d2h_pinned_opts(src, dst, &DmaOptions::default())
    }

    /// [`Device::memcpy_d2h_pinned`] with explicit [`DmaOptions`].
    pub fn memcpy_d2h_pinned_opts(
        &self,
        src: u64,
        dst: &mut PinnedBuffer<'_>,
        opts: &DmaOptions,
    ) -> Result<()> {
        self.pinned_transfer(dst, |device, host, len, pending| {
            device.push_dma_read_staged(host, len, src, opts, pending)?;
            let count = pending.len();
            device.await_dma(pending, count, opts.timeout, "dma-readlist")
        })
    }

    /// Run `transfer` over the host mapping of `buffer`, after checking that
    /// the buffer belongs to this device. Commands left uncollected by a
    /// failure are abandoned; if DMA may still be in flight, the mapping is
    /// leaked and the buffer emptied, as for the staging buffers.
    fn pinned_transfer<F>(&self, buffer: &mut PinnedBuffer<'_>, transfer: F) -> Result<()>
    where
        F: FnOnce(&Self, &dyn DmaHostBuffer, usize, &mut Vec<u16>) -> Result<()>,
    {
        if !std::ptr::eq(buffer.device, self as *const Self as *const ()) {
            return Err(Error::Limit(
                "pinned buffer was allocated by a different device".into(),
            ));
        }
        let Some(host) = buffer.host.as_deref() else {
            return Ok(());
        };
        if buffer.len == 0 {
            return Ok(());
        }
        let mut pending = Vec::new();
        let result = transfer(self, host, buffer.len, &mut pending);
        let uncollected = !pending.is_empty();
        self.abandon_all(&mut pending);
        if let Err(e) = &result
            && (uncollected || dma_may_be_in_flight(e))
        {
            if let Some(host) = buffer.host.take() {
                std::mem::forget(host);
            }
            buffer.len = 0;
        }
        result
    }

    /// Copy `src.len()` bytes from host memory to device address `dst` via a DMA
    /// write-list command, splitting the transfer to honour the device's DMA
    /// element-size and element-count limits. The data is staged through the
    /// device's persistent DMA host buffers.
    ///
    /// Large transfers are pipelined so that the host copy of each chunk
    /// overlaps the DMA of the previous one, and the staging buffers are leaked
    /// if a command fails in flight; see [`Device::memcpy_d2h`].
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
        self.h2d_staged(src.len(), dst, opts, |staged, offset| {
            staged.copy_from_slice(&src[offset..offset + staged.len()]);
        })
    }

    /// Write `total` bytes to device address `dst` through the persistent
    /// staging buffers, pipelined as described for [`Device::memcpy_h2d`].
    /// Before each chunk is sent, `produce` fills the staged bytes with the data
    /// destined for byte offset `offset` of the transfer; the staged slice
    /// length is the chunk length.
    pub(crate) fn h2d_staged<F>(
        &self,
        total: usize,
        dst: u64,
        opts: &DmaOptions,
        produce: F,
    ) -> Result<()>
    where
        F: FnMut(&mut [u8], usize),
    {
        if total == 0 {
            return Ok(());
        }
        let mut buffers = self.take_staging(total)?;
        let mut pending = Vec::new();
        let result = self.h2d_pipeline(&mut buffers, total, dst, opts, produce, &mut pending);
        self.restore_staging(&mut buffers, &mut pending, &result);
        result
    }

    /// Extract the compute-minion trace buffer (`TRACE_BUFFER_CM`).
    pub fn extract_cm_trace(&self) -> Result<Vec<u8>> {
        self.extract_trace(ops::trace_buffer_type::TRACE_BUFFER_CM as u8)
    }

    /// Extract a device trace buffer of the given `trace_buffer_type`.
    pub fn extract_trace(&self, trace_type: u8) -> Result<Vec<u8>> {
        self.transport.extract_trace(trace_type)
    }

    /// Reset the compute minions on the shires in `shire_mask`.
    ///
    /// Sends a `CM_RESET_CMD` and blocks until the firmware acknowledges it.
    /// The device node remains open; DMA and host state are unaffected. This
    /// is the appropriate response to a kernel that has hung inside the RISC-V
    /// compute shires without corrupting firmware state.
    ///
    /// On success the reset shires are ready for a new kernel launch. Any
    /// previously loaded kernels remain mapped (the DRAM allocator state is
    /// not modified); loaded kernels can be launched again immediately.
    ///
    /// Returns [`Error::Device`] with `command = "cm-reset"` if the firmware
    /// rejects the request (e.g. an invalid `shire_mask`).
    pub fn reset_shires(&self, shire_mask: u64) -> Result<()> {
        let tag = self.next_tag()?;
        let cmd = proto::build_cm_reset(tag, shire_mask);
        // CM reset goes through the high-priority SQ (HPSQ), which is the MM
        // firmware's management path. The standard SQ (flags=0) is not
        // monitored for management commands on this firmware version; submitting
        // there produces no response. CMD_DESC_FLAG_HIGH_PRIORITY routes the
        // command to the HPSQ. The `collect_response` caller uses
        // `GET_CQ_AVAIL_BITMAP` to find whichever CQ the response lands on.
        let timeout = Some(self.default_timeout.get());
        let rsp = self.submit(0, &cmd, desc_flags::HIGH_PRIORITY, tag, timeout)?;
        let status = proto::cm_reset_response_status(&rsp.bytes)
            .ok_or_else(|| Error::Protocol("CM reset response truncated".into()))?;
        if status != ops::DEV_OPS_API_CM_RESET_RESPONSE::DEV_OPS_API_CM_RESET_RESPONSE_SUCCESS {
            return Err(Error::Device {
                command: "cm-reset",
                code: status,
            });
        }
        Ok(())
    }

    // --- internals ---

    /// Remove the persistent staging buffers from the device for one transfer
    /// of `total` bytes, returning one buffer if the transfer fits in half the
    /// staging capacity and two (for pipelining) otherwise. Each buffer used
    /// must hold at least `min(total, capacity / 2)` bytes; one that does not
    /// is replaced by a buffer sized to the next power of two, bounded by half
    /// the capacity, so that a sequence of growing transfers reallocates only
    /// logarithmically often. Buffers beyond those needed are carried along
    /// unchanged. The caller hands them back via [`Device::restore_staging`].
    fn take_staging(&self, total: usize) -> Result<Vec<Box<dyn DmaHostBuffer>>> {
        let half = self.staging_capacity.get() / 2;
        let required = total.min(half);
        let needed = if total > half { 2 } else { 1 };
        let mut buffers = std::mem::take(&mut *self.staging.borrow_mut());
        for index in 0..needed {
            if buffers
                .get(index)
                .is_some_and(|b| b.as_slice().len() >= required)
            {
                continue;
            }
            if index < buffers.len() {
                // Unmap the undersized buffer before mapping its replacement,
                // so that both are never held at once.
                drop(buffers.remove(index));
            }
            let replacement = match self
                .transport
                .dma_host_buffer(required.next_power_of_two().min(half))
            {
                Ok(buffer) => buffer,
                Err(e) => {
                    self.staging.replace(buffers);
                    return Err(e);
                }
            };
            buffers.insert(index, replacement);
        }
        Ok(buffers)
    }

    /// Return the staging buffers to the device after a transfer completed
    /// with `result`. Commands still in `pending` were pushed but never
    /// collected, so they are abandoned. If any DMA may still be in flight the
    /// buffers are leaked, so the device cannot write into memory the driver
    /// has reallocated; otherwise they are retained for reuse.
    fn restore_staging(
        &self,
        buffers: &mut Vec<Box<dyn DmaHostBuffer>>,
        pending: &mut Vec<u16>,
        result: &Result<()>,
    ) {
        let uncollected = !pending.is_empty();
        for tag in pending.drain(..) {
            self.abandon(tag);
        }
        let buffers = std::mem::take(buffers);
        match result {
            Err(e) if uncollected || dma_may_be_in_flight(e) => {
                for buffer in buffers {
                    std::mem::forget(buffer);
                }
            }
            _ => {
                self.staging.replace(buffers);
            }
        }
    }

    /// Write `total` bytes to device address `dst`, cycling through `buffers`
    /// in chunks of the smallest buffer's length. The chunk staged in one
    /// buffer is pushed before the DMA of the chunk staged in the other is
    /// awaited, so the host copy of each chunk overlaps the transfer of its
    /// predecessor. Tags of pushed but uncollected commands are kept in
    /// `pending` for the caller to abandon on failure.
    fn h2d_pipeline<F>(
        &self,
        buffers: &mut [Box<dyn DmaHostBuffer>],
        total: usize,
        dst: u64,
        opts: &DmaOptions,
        mut produce: F,
        pending: &mut Vec<u16>,
    ) -> Result<()>
    where
        F: FnMut(&mut [u8], usize),
    {
        let chunk = staging_chunk(buffers);
        let mut previous_commands = 0;
        for (index, offset) in (0..total).step_by(chunk).enumerate() {
            let len = chunk.min(total - offset);
            let buffer = &mut buffers[index % buffers.len()];
            produce(&mut buffer.as_mut_slice()[..len], offset);
            self.push_dma_write_staged(&**buffer, len, dst + offset as u64, opts, pending)?;
            // The buffer refilled on the next iteration is the one whose DMA is
            // awaited here.
            self.await_dma(pending, previous_commands, opts.timeout, "dma-writelist")?;
            previous_commands = pending.len();
        }
        self.await_dma(pending, previous_commands, opts.timeout, "dma-writelist")
    }

    /// Read `dst.len()` bytes from device address `src`, cycling through
    /// `buffers` in chunks of the smallest buffer's length. The DMA of the next
    /// chunk is pushed before the current chunk is awaited and copied out, so
    /// the copy overlaps the next transfer. Tags of pushed but uncollected
    /// commands are kept in `pending` for the caller to abandon on failure.
    fn d2h_pipeline(
        &self,
        buffers: &[Box<dyn DmaHostBuffer>],
        src: u64,
        dst: &mut [u8],
        opts: &DmaOptions,
        pending: &mut Vec<u16>,
    ) -> Result<()> {
        let total = dst.len();
        let chunk = staging_chunk(buffers);
        let chunk_count = total.div_ceil(chunk);
        let read_chunk = |index: usize, pending: &mut Vec<u16>| {
            let offset = index * chunk;
            let len = chunk.min(total - offset);
            let buffer = &*buffers[index % buffers.len()];
            self.push_dma_read_staged(buffer, len, src + offset as u64, opts, pending)
        };
        read_chunk(0, pending)?;
        for (index, out) in dst.chunks_mut(chunk).enumerate() {
            let current_commands = pending.len();
            if index + 1 < chunk_count {
                // The next chunk's buffer was emptied on the previous iteration.
                read_chunk(index + 1, pending)?;
            }
            self.await_dma(pending, current_commands, opts.timeout, "dma-readlist")?;
            out.copy_from_slice(&buffers[index % buffers.len()].as_slice()[..out.len()]);
        }
        Ok(())
    }

    /// Push, without awaiting, the DMA read-list commands transferring `len`
    /// bytes from device address `src` into the first `len` bytes of `host`,
    /// split into list nodes and commands according to the DMA limits. Each
    /// command carries `BARRIER`; its tag is appended to `pending`.
    fn push_dma_read_staged(
        &self,
        host: &dyn DmaHostBuffer,
        len: usize,
        src: u64,
        opts: &DmaOptions,
        pending: &mut Vec<u16>,
    ) -> Result<()> {
        let max_elem = (self.dram.dma_max_elem_size as usize).max(1);
        let max_nodes = self.max_list_nodes()?;
        let hvirt = host.virt_addr();
        let hphys = host.phys_addr();
        let mut offset = 0usize;
        let mut nodes: Vec<proto::DmaReadNode> = Vec::with_capacity(max_nodes);
        while offset < len {
            let size = (len - offset).min(max_elem);
            nodes.push(proto::DmaReadNode {
                dst_host_virt_addr: hvirt + offset as u64,
                dst_host_phy_addr: node_phys(hphys, offset),
                src_device_phy_addr: src + offset as u64,
                size: size as u32,
                _pad: [0; 4],
            });
            offset += size;
            if nodes.len() == max_nodes || offset >= len {
                let tag = self.next_tag()?;
                let cmd = proto::build_dma_readlist(tag, cmd_flags::BARRIER, &nodes);
                self.push_cmd(opts.sq_index, &cmd, desc_flags::DMA, tag)?;
                pending.push(tag);
                nodes.clear();
            }
        }
        Ok(())
    }

    /// Push the DMA write-list commands transferring the first `len` bytes of
    /// `host` to device address `dst`; the counterpart of
    /// [`Device::push_dma_read_staged`].
    fn push_dma_write_staged(
        &self,
        host: &dyn DmaHostBuffer,
        len: usize,
        dst: u64,
        opts: &DmaOptions,
        pending: &mut Vec<u16>,
    ) -> Result<()> {
        let max_elem = (self.dram.dma_max_elem_size as usize).max(1);
        let max_nodes = self.max_list_nodes()?;
        let hvirt = host.virt_addr();
        let hphys = host.phys_addr();
        let mut offset = 0usize;
        let mut nodes: Vec<proto::DmaWriteNode> = Vec::with_capacity(max_nodes);
        while offset < len {
            let size = (len - offset).min(max_elem);
            nodes.push(proto::DmaWriteNode {
                src_host_virt_addr: hvirt + offset as u64,
                src_host_phy_addr: node_phys(hphys, offset),
                dst_device_phy_addr: dst + offset as u64,
                size: size as u32,
                _pad: [0; 4],
            });
            offset += size;
            if nodes.len() == max_nodes || offset >= len {
                let tag = self.next_tag()?;
                let cmd = proto::build_dma_writelist(tag, cmd_flags::BARRIER, &nodes);
                self.push_cmd(opts.sq_index, &cmd, desc_flags::DMA, tag)?;
                pending.push(tag);
                nodes.clear();
            }
        }
        Ok(())
    }

    /// Collect the completions of the first `count` DMA commands in `pending`,
    /// oldest first, removing each from `pending` before it is awaited, and
    /// check that each reports success. `command` names the command kind in
    /// the error returned for a failing status.
    fn await_dma(
        &self,
        pending: &mut Vec<u16>,
        count: usize,
        timeout: Option<Duration>,
        command: &'static str,
    ) -> Result<()> {
        for tag in pending.drain(..count).collect::<Vec<_>>() {
            let rsp = self.collect_response(tag, timeout, "DMA completion")?;
            let status = proto::response_status(&rsp.bytes)
                .ok_or_else(|| Error::Protocol(format!("{command} response truncated")))?;
            if status != ops::DEV_OPS_API_DMA_RESPONSE::DEV_OPS_API_DMA_RESPONSE_COMPLETE {
                return Err(Error::Device {
                    command,
                    code: status,
                });
            }
        }
        Ok(())
    }

    /// Largest number of DMA list nodes a single command may carry: the least of
    /// the device's element-count limit, the count representable in the `u16`
    /// command-size field, and the count fitting one submission-queue message.
    fn max_list_nodes(&self) -> Result<usize> {
        if let Some(n) = self.max_list_nodes.get() {
            return Ok(n);
        }
        let node_size = core::mem::size_of::<proto::DmaReadNode>();
        debug_assert_eq!(node_size, core::mem::size_of::<proto::DmaWriteNode>());
        let by_field = (u16::MAX as usize - proto::CMN_HEADER_SIZE) / node_size;
        let msg = self.transport.sq_max_msg_size()? as usize;
        let by_message = msg.saturating_sub(proto::CMN_HEADER_SIZE) / node_size;
        let n = (self.dram.dma_max_elem_count as usize)
            .min(by_field)
            .min(by_message)
            .max(1);
        self.max_list_nodes.set(Some(n));
        Ok(n)
    }

    /// Push a command bearing the reserved `tag` onto the submission queue,
    /// retrying until space is available. On failure the command was never
    /// queued, so the tag is released.
    fn push_cmd(&self, sq_index: u16, cmd: &[u8], desc_flags: u8, tag: u16) -> Result<()> {
        let pushed = self.push_cmd_inner(sq_index, cmd, desc_flags);
        if pushed.is_err() {
            self.outstanding.borrow_mut().remove(&tag);
        }
        pushed
    }

    fn push_cmd_inner(&self, sq_index: u16, cmd: &[u8], desc_flags: u8) -> Result<()> {
        // Longest a single `wait_sq` blocks before re-polling. A backend whose
        // wait returns immediately (the emulator does) must not be mistaken for
        // a genuine timeout; the deadline is the sole authority on giving up.
        let limit = self.default_timeout.get();
        let slice = Duration::from_millis(250);
        let deadline = Instant::now() + limit;
        let mut woken = false;
        loop {
            if self.transport.push_sq(sq_index, cmd, desc_flags)? {
                return Ok(());
            }
            if woken {
                // Readiness was reported but the push still found no space:
                // back off rather than spin on a persistently ready queue.
                thread::sleep(Duration::from_millis(1));
            }
            if Instant::now() >= deadline {
                return Err(Error::Timeout {
                    operation: "submission-queue space",
                    limit,
                });
            }
            woken = self.transport.wait_sq(remaining(deadline).min(slice))?;
            if !woken {
                thread::sleep(Duration::from_millis(1));
            }
        }
    }

    /// Block until the CQ response bearing `expected_tag` arrives.
    ///
    /// `operation` names the blocking operation for diagnostic purposes (e.g.
    /// `"kernel completion"`, `"DMA completion"`). Responses for other
    /// outstanding tags are placed in `self.stash` so that concurrent
    /// `launch_async` / `wait_launch` sequences do not discard each other's
    /// completions; see [`Device::park`] for the handling of other tags.
    ///
    /// On success the tag is released. On any error the tag is marked
    /// abandoned: it stays reserved until its late response (if any) arrives
    /// and is discarded, so it can never be confused with a newer command.
    ///
    /// `timeout` is `None` for unlimited waits (DMA default) or `Some(d)` to
    /// surface an [`Error::Timeout`] after `d` elapses.
    fn collect_response(
        &self,
        expected_tag: u16,
        timeout: Option<Duration>,
        operation: &'static str,
    ) -> Result<PoppedResponse> {
        let collected = self.collect_response_inner(expected_tag, timeout, operation);
        match collected {
            Ok(_) => {
                self.outstanding.borrow_mut().remove(&expected_tag);
            }
            Err(_) => {
                self.abandoned.borrow_mut().insert(expected_tag);
            }
        }
        collected
    }

    fn collect_response_inner(
        &self,
        expected_tag: u16,
        timeout: Option<Duration>,
        operation: &'static str,
    ) -> Result<PoppedResponse> {
        let slice = Duration::from_millis(250);
        let deadline = timeout.map(|t| Instant::now() + t);

        // Check the stash before polling the CQ: if a prior `collect_response`
        // parked this tag, return it immediately without touching the hardware.
        if let Some(rsp) = self.stash.borrow_mut().remove(&expected_tag) {
            return Ok(rsp);
        }

        let mut woken = false;
        loop {
            if let Some(rsp) = self.transport.pop_cq()? {
                woken = false;
                match proto::ResponseHeader::parse(&rsp.bytes) {
                    Some(hdr) if hdr.tag_id == expected_tag => return Ok(rsp),
                    Some(hdr) => self.park(hdr.tag_id, rsp),
                    None => {
                        return Err(Error::Protocol(format!(
                            "malformed completion ({} bytes) while awaiting {operation}",
                            rsp.bytes.len()
                        )));
                    }
                }
                continue;
            }
            if woken {
                // Readiness was reported but nothing was popped (for example, a
                // completion on a queue this handle does not drain): back off
                // rather than spin.
                thread::sleep(Duration::from_millis(1));
            }
            let wait = match deadline {
                Some(dl) => {
                    if Instant::now() >= dl {
                        return Err(Error::Timeout {
                            operation,
                            limit: timeout.unwrap_or_default(),
                        });
                    }
                    remaining(dl).min(slice)
                }
                None => slice,
            };
            // A false return means no completion arrived in this slice; keep
            // polling until the deadline.
            woken = self.transport.wait_cq(wait)?;
            if !woken {
                thread::sleep(Duration::from_millis(1));
            }
        }
    }

    /// Dispose of a response for `tag` popped whilst awaiting another tag.
    ///
    /// A response for an abandoned tag is discarded and the tag released; one
    /// for an outstanding tag is stashed for its own waiter; one for a tag this
    /// device never issued (or has already completed) is discarded, since no
    /// waiter could ever claim it and it would otherwise shadow a future
    /// command reusing the tag.
    fn park(&self, tag: u16, rsp: PoppedResponse) {
        if self.abandoned.borrow_mut().remove(&tag) {
            self.outstanding.borrow_mut().remove(&tag);
        } else if self.outstanding.borrow().contains(&tag) {
            self.stash.borrow_mut().insert(tag, rsp);
        }
    }

    /// Give up on the outstanding command `tag` without awaiting it. A response
    /// already stashed is discarded and the tag released; otherwise the tag is
    /// marked abandoned, so its late response is discarded on arrival.
    fn abandon(&self, tag: u16) {
        if self.stash.borrow_mut().remove(&tag).is_some() {
            self.outstanding.borrow_mut().remove(&tag);
        } else {
            self.abandoned.borrow_mut().insert(tag);
        }
    }

    /// Abandon every tag in `tags`, leaving it empty.
    fn abandon_all(&self, tags: &mut Vec<u16>) {
        for tag in tags.drain(..) {
            self.abandon(tag);
        }
    }

    /// Copy `args` into the host buffer of argument slot `slot`, mapping the
    /// buffer on first use, and push, without awaiting, the DMA commands
    /// writing them to the slot's device region on submission queue
    /// `sq_index`. Returns the device address of the region. Tags of the
    /// pushed commands are appended to `pending`, and the slot is assigned to
    /// the last of them: each command carries `BARRIER`, so its completion
    /// implies that of the others.
    fn push_args_dma(
        &self,
        slot: usize,
        args: &[u8],
        sq_index: u16,
        pending: &mut Vec<u16>,
    ) -> Result<u64> {
        let (region, host) = {
            let mut slots = self.args_slots.borrow_mut();
            (slots[slot].region, slots[slot].host.take())
        };
        let mut host = match host {
            Some(host) => host,
            // The region was allocated for at least `args.len()` bytes, so its
            // size fits the host address space.
            None => self.transport.dma_host_buffer(region.size as usize)?,
        };
        host.as_mut_slice()[..args.len()].copy_from_slice(args);
        let opts = DmaOptions::new().on_sq(sq_index);
        let pushed = self.push_dma_write_staged(&*host, args.len(), region.addr, &opts, pending);
        let mut slots = self.args_slots.borrow_mut();
        slots[slot].host = Some(host);
        if let Some(&tag) = pending.last() {
            slots[slot].owner = Some(tag);
        }
        pushed.map(|()| region.addr)
    }

    /// Push a command and block for the response bearing `expected_tag`.
    ///
    /// A thin wrapper around [`push_cmd`] + [`collect_response`]; used for
    /// operations that are inherently synchronous (DMA commands). Kernel launches
    /// use [`Device::launch_async`] + [`Device::wait_launch`] so they can carry
    /// a per-launch timeout.
    fn submit(
        &self,
        sq_index: u16,
        cmd: &[u8],
        desc_flags: u8,
        expected_tag: u16,
        timeout: Option<Duration>,
    ) -> Result<PoppedResponse> {
        self.push_cmd(sq_index, cmd, desc_flags, expected_tag)?;
        self.collect_response(expected_tag, timeout, "DMA completion")
    }

    /// Reserve a command tag not currently outstanding. The tag stays reserved
    /// until its response is collected, or, if its waiter gave up, until the
    /// late response arrives. Returns [`Error::Limit`] if all 65 536 tags are
    /// reserved.
    fn next_tag(&self) -> Result<u16> {
        let mut outstanding = self.outstanding.borrow_mut();
        let start = self.tag.get();
        let mut candidate = start;
        loop {
            if outstanding.insert(candidate) {
                self.tag.set(candidate.wrapping_add(1));
                return Ok(candidate);
            }
            candidate = candidate.wrapping_add(1);
            if candidate == start {
                return Err(Error::Limit(
                    "all 65536 command tags are outstanding; collect pending launches".into(),
                ));
            }
        }
    }
}

/// Whether a failed DMA command may have left the device still accessing the
/// staging buffer: the command may have been queued (timeout, transport or
/// protocol failure), as opposed to a definite completion with an error status.
fn dma_may_be_in_flight(error: &Error) -> bool {
    matches!(
        error,
        Error::Timeout { .. } | Error::Io { .. } | Error::Protocol(_)
    )
}

/// Physical address for a DMA node at `offset` into a staging buffer whose base
/// physical address is `base`. A zero base means the backend resolves the
/// physical address itself, so it stays zero.
fn node_phys(base: u64, offset: usize) -> u64 {
    if base == 0 { 0 } else { base + offset as u64 }
}

/// Round a requested staging capacity down to a multiple of twice the device's
/// DMA alignment, with a minimum of one such quantum, so that each half is a
/// whole number of alignment quanta.
fn align_staging_capacity(bytes: usize, dram: &DramInfo) -> usize {
    let quantum = 2 * dram.alignment() as usize;
    (bytes / quantum).max(1) * quantum
}

/// Chunk length for a pipelined transfer through `buffers`: the length of the
/// smallest, so that every chunk fits whichever buffer it is staged in.
fn staging_chunk(buffers: &[Box<dyn DmaHostBuffer>]) -> usize {
    buffers
        .iter()
        .map(|buffer| buffer.as_slice().len())
        .min()
        .unwrap_or(1)
        .max(1)
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
