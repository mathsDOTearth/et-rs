//! The [`Transport`] abstraction and its concrete backends.
//!
//! A [`Transport`] hides the mechanism by which host commands reach the device.
//! The shipped backend, [`IoctlTransport`], drives the PCIe kernel driver's
//! character device (`/dev/etN_ops`) via the `ETSOC1_IOCTL_*` calls and only
//! works against real hardware. The software emulator bundled with the SDK is
//! reachable solely through the vendor C++ device-layer over a private IPC, not
//! through these ioctls; the trait exists so that such an alternative backend
//! (for example an FFI shim over `libdeviceLayer`) can be added later without
//! disturbing the [`crate::Device`] API layered on top.

use crate::error::Result;
use std::time::Duration;

mod ioctl_transport;
pub use ioctl_transport::IoctlTransport;

#[cfg(feature = "emu")]
mod ffi_transport;
#[cfg(feature = "emu")]
pub use ffi_transport::FfiTransport;

/// Host-visible description of the device's user-accessible DRAM region and its
/// DMA constraints, from `ETSOC1_IOCTL_GET_USER_DRAM_INFO`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DramInfo {
    /// Base device physical address of the user DRAM region.
    pub base: u64,
    /// Size of the region in bytes.
    pub size: u64,
    /// Maximum size of a single DMA element/node in bytes.
    pub dma_max_elem_size: u32,
    /// Maximum number of elements/nodes in a single DMA-list command.
    pub dma_max_elem_count: u16,
    /// Required address-alignment quantum in bytes, as reported by the device
    /// (e.g. 64 for a cache line). Despite the driver field being named
    /// `align_in_bits`, hardware and emulator both report a byte count here, not
    /// a log2 bit-count.
    pub dma_alignment: u16,
}

impl DramInfo {
    /// The address alignment to apply, in bytes: the device's reported quantum
    /// rounded up to a power of two (at least 1). Over-alignment is always safe,
    /// and the `u16` source keeps this bounded, so it can never swallow the DRAM
    /// region the way a mis-scaled shift could.
    pub fn alignment(&self) -> u64 {
        (self.dma_alignment as u64).max(1).next_power_of_two()
    }
}

/// A host-side buffer usable as a DMA endpoint by a particular backend.
///
/// DMA nodes carry both a host *virtual* address and a host *physical* address.
/// A pinning kernel driver resolves the physical address itself (so it may be
/// left 0), whereas the software emulator dereferences the physical field
/// directly and therefore needs a real, backend-provided address. This
/// abstraction lets [`crate::Device`] stage DMA transfers through memory the
/// active transport can actually reach.
pub trait DmaHostBuffer {
    /// The staging bytes, for writing before a host-to-device transfer.
    fn as_mut_slice(&mut self) -> &mut [u8];
    /// The staging bytes, for reading after a device-to-host transfer.
    fn as_slice(&self) -> &[u8];
    /// Host virtual address to place in a DMA node's `*_host_virt_addr` field.
    fn virt_addr(&self) -> u64;
    /// Host physical address for the `*_host_phy_addr` field (0 if the backend
    /// resolves it, e.g. a pinning kernel driver).
    fn phys_addr(&self) -> u64;
}

/// Default [`DmaHostBuffer`] backed by an owned `Vec`, for backends (the kernel
/// driver) that accept any host virtual address and resolve the physical
/// address themselves.
pub struct VecDmaBuffer {
    buf: Vec<u8>,
}

impl VecDmaBuffer {
    /// Allocate a zeroed staging buffer of `size` bytes.
    pub fn new(size: usize) -> Self {
        VecDmaBuffer {
            buf: vec![0u8; size],
        }
    }
}

impl DmaHostBuffer for VecDmaBuffer {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.buf
    }
    fn as_slice(&self) -> &[u8] {
        &self.buf
    }
    fn virt_addr(&self) -> u64 {
        self.buf.as_ptr() as u64
    }
    fn phys_addr(&self) -> u64 {
        0
    }
}

/// A single response drained from a completion queue.
#[derive(Clone, Debug)]
pub struct PoppedResponse {
    /// The raw response bytes (common header followed by the message body).
    pub bytes: Vec<u8>,
    /// The completion queue the response was drawn from.
    pub cq_index: u16,
}

/// Device configuration queried from the device: the compute-shire mask and
/// cache-line size. [`crate::Device::topology`] combines it with architectural
/// constants into a [`crate::Topology`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceConfig {
    /// Bitmask of compute shires present/enabled on the device.
    pub shire_mask: u64,
    /// Cache line size in bytes.
    pub cache_line: u32,
}

/// Full device properties returned by `ETSOC1_IOCTL_GET_DEVICE_CONFIGURATION`.
///
/// All thirteen fields of the driver's `dev_config` descriptor are exposed here.
/// Obtain this via [`crate::Device::properties`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceProperties {
    /// Total L3 cache size, in KB.
    pub total_l3_size:      u32,
    /// Total L2 cache size (summed across all shires), in KB.
    pub total_l2_size:      u32,
    /// Total Minion L1 scratchpad size (summed across all Minions), in KB.
    pub total_scp_size:     u32,
    /// DDR bandwidth, in MB/s.
    pub ddr_bandwidth:      u32,
    /// Minion boot frequency, in MHz. Divide a cycle-count delta by this
    /// value (multiplied by 1e6) to obtain an elapsed time in seconds.
    pub minion_boot_freq:   u32,
    /// Bitmask of compute shires present/enabled on the device.
    pub shire_mask:         u32,
    /// Form-factor code (see `dev_config_form_factor` in the uapi header).
    pub form_factor:        u8,
    /// Thermal design power, in watts.
    pub tdp:                u8,
    /// Cache-line size, in bytes (typically 64).
    pub cache_line_size:    u8,
    /// Number of L2 cache banks per shire.
    pub num_l2_cache_banks: u8,
    /// Shire ID of the spare/sync Minion shire.
    pub sync_min_shire_id:  u8,
    /// Architecture revision code (see `dev_config_arch_revision`).
    pub arch_rev:           u8,
    /// Physical device node index (the `N` in `/dev/etN_ops`).
    pub devnum:             u8,
}

/// The low-level command channel to a single ET-SoC-1 device.
///
/// Implementations must be usable from a single thread; the ET device model
/// guarantees thread safety only between one master-minion thread and one
/// service-processor thread, so no interior synchronisation is assumed here.
pub trait Transport {
    /// Query the user DRAM region geometry and DMA limits.
    fn dram_info(&self) -> Result<DramInfo>;

    /// Query the device configuration (compute-shire mask and cache geometry).
    ///
    /// The default returns a conservative single-shire configuration (shire 0,
    /// 64-byte cache line); backends that can query the real device override it.
    fn device_config(&self) -> Result<DeviceConfig> {
        Ok(DeviceConfig {
            shire_mask: 0x1,
            cache_line: 64,
        })
    }

    /// Query the full device properties (all fields of `dev_config`).
    ///
    /// The default derives from [`Transport::device_config`]; backends that
    /// query real hardware override both together. Most numerical fields
    /// default to zero when the transport cannot provide them.
    fn device_properties(&self) -> Result<DeviceProperties> {
        let cfg = self.device_config()?;
        Ok(DeviceProperties {
            total_l3_size:      0,
            total_l2_size:      0,
            total_scp_size:     0,
            ddr_bandwidth:      0,
            minion_boot_freq:   0,
            shire_mask:         cfg.shire_mask as u32,
            form_factor:        0,
            tdp:                0,
            cache_line_size:    cfg.cache_line as u8,
            num_l2_cache_banks: 0,
            sync_min_shire_id:  0,
            arch_rev:           0,
            devnum:             0,
        })
    }

    /// Push a firmware/kernel image to the device (`ETSOC1_IOCTL_FW_UPDATE`).
    fn fw_update(&self, image: &[u8]) -> Result<()>;

    /// Number of master-minion submission queues.
    fn sq_count(&self) -> Result<u16>;

    /// Maximum command message size, in bytes, accepted by a submission queue.
    fn sq_max_msg_size(&self) -> Result<u16>;

    /// Push a command onto submission queue `sq_index` (`ETSOC1_IOCTL_PUSH_SQ`).
    ///
    /// `flags` carries [`crate::proto::desc_flags`] bits. Returns `Ok(false)` if
    /// the queue was full and the caller should retry after [`Transport::wait_sq`].
    fn push_sq(&self, sq_index: u16, cmd: &[u8], flags: u8) -> Result<bool>;

    /// Pop one response from the completion queue (`ETSOC1_IOCTL_POP_CQ`).
    ///
    /// Returns `Ok(None)` when no response is currently available.
    fn pop_cq(&self) -> Result<Option<PoppedResponse>>;

    /// Extract a device trace buffer of the given `trace_buffer_type`
    /// (`ETSOC1_IOCTL_EXTRACT_TRACE_BUFFER`), sized from
    /// `ETSOC1_IOCTL_GET_TRACE_BUFFER_SIZE`.
    fn extract_trace(&self, trace_type: u8) -> Result<Vec<u8>>;

    /// Block until the completion queue is readable, or `timeout` elapses.
    ///
    /// Returns `true` if it became readable. The default assumes immediate
    /// readiness, which suits synchronous or in-memory backends.
    fn wait_cq(&self, _timeout: Duration) -> Result<bool> {
        Ok(true)
    }

    /// Block until some submission queue has free space, or `timeout` elapses.
    ///
    /// Returns `true` if space became available. The default assumes immediate
    /// readiness.
    fn wait_sq(&self, _timeout: Duration) -> Result<bool> {
        Ok(true)
    }

    /// Allocate a host buffer usable as a DMA endpoint for this backend.
    ///
    /// The default returns a plain heap buffer with a zero physical address,
    /// correct for a pinning kernel driver. Backends that require registered DMA
    /// memory (the emulator) override this.
    fn dma_host_buffer(&self, size: usize) -> Result<Box<dyn DmaHostBuffer>> {
        Ok(Box::new(VecDmaBuffer::new(size)))
    }
}
