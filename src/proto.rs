//! Device-ops RPC wire format: command builders and response parsers.
//!
//! The ET-SoC-1 host/device message structs (device_apis_message_types.h,
//! device_ops_api_rpc_types.h) are declared `packed, aligned(8)` with fields
//! ordered by decreasing size, so every field is already naturally aligned and
//! the `packed` attribute changes no offsets. They are transcribed here as
//! `#[repr(C)]` types whose layouts are checked against the ABI at compile time,
//! because bindgen 0.72 miscompiles the `packed`+`aligned` combination (see
//! `build.rs`). All multi-byte integers are little-endian on both the host and
//! the RISC-V device, so the in-memory representation is the wire representation.
//!
//! Message identifiers and flag values are taken from the bindgen-generated
//! [`crate::ffi::ops`] enumerations, keeping this module in step with the SDK.

/// Command-header flag bits (`enum CMD_FLAGS`), placed in [`CmnHeader::flags`].
pub mod cmd_flags {
    use crate::ffi::ops::CMD_FLAGS;

    /// Drain all prior commands before this one executes.
    pub const BARRIER: u16 = CMD_FLAGS::CMD_FLAGS_BARRIER_ENABLE as u16;
    /// A U-mode (compute-kernel) trace configuration is present in the payload.
    pub const COMPUTE_KERNEL_TRACE: u16 = CMD_FLAGS::CMD_FLAGS_COMPUTE_KERNEL_TRACE_ENABLE as u16;
    /// Flush the L3 cache before launching the kernel.
    pub const FLUSH_L3: u16 = CMD_FLAGS::CMD_FLAGS_KERNEL_LAUNCH_FLUSH_L3 as u16;
    /// User kernel arguments are embedded in the launch payload.
    pub const KERNEL_ARGS_EMBEDDED: u16 = CMD_FLAGS::CMD_FLAGS_KERNEL_LAUNCH_ARGS_EMBEDDED as u16;
    /// A U-mode stack configuration is present in the launch payload.
    pub const USER_STACK_CFG: u16 = CMD_FLAGS::CMD_FLAGS_KERNEL_LAUNCH_USER_STACK_CFG as u16;
}

/// Driver-level command-descriptor flag bits (`enum cmd_desc_flag`), placed in
/// `cmd_desc.flags` when pushing to a submission queue.
pub mod desc_flags {
    use crate::ffi::ops::cmd_desc_flag;

    /// The command carries host DMA addresses that the driver must translate.
    pub const DMA: u8 = cmd_desc_flag::CMD_DESC_FLAG_DMA as u8;
    /// High-priority submission queue.
    pub const HIGH_PRIORITY: u8 = cmd_desc_flag::CMD_DESC_FLAG_HIGH_PRIORITY as u8;
    /// The command carries peer-to-peer DMA addresses.
    pub const P2PDMA: u8 = cmd_desc_flag::CMD_DESC_FLAG_P2PDMA as u8;
}

/// Message identifiers used by the commands this crate builds.
pub mod msg_id {
    use crate::ffi::ops::device_ops_api_msg_e as m;

    pub const KERNEL_LAUNCH_CMD: u16 = m::DEV_OPS_API_MID_DEVICE_OPS_KERNEL_LAUNCH_CMD as u16;
    pub const KERNEL_LAUNCH_RSP: u16 = m::DEV_OPS_API_MID_DEVICE_OPS_KERNEL_LAUNCH_RSP as u16;
    pub const DMA_READLIST_CMD: u16 = m::DEV_OPS_API_MID_DEVICE_OPS_DMA_READLIST_CMD as u16;
    pub const DMA_READLIST_RSP: u16 = m::DEV_OPS_API_MID_DEVICE_OPS_DMA_READLIST_RSP as u16;
    pub const DMA_WRITELIST_CMD: u16 = m::DEV_OPS_API_MID_DEVICE_OPS_DMA_WRITELIST_CMD as u16;
    pub const DMA_WRITELIST_RSP: u16 = m::DEV_OPS_API_MID_DEVICE_OPS_DMA_WRITELIST_RSP as u16;
}

/// Size in bytes of the common message header ([`CmnHeader`]).
pub const CMN_HEADER_SIZE: usize = 8;

/// Common message header (`struct cmn_header_t`), shared by commands, responses
/// and events. Eight bytes, 64-bit aligned in the C ABI.
///
/// Despite the SDK header describing `size` as "the payload that follows the
/// message header", the device firmware and device-layer validate it against
/// the *total* command length (header included): a command whose `size` does not
/// equal the number of bytes submitted is rejected. This crate therefore always
/// sets `size` to the whole command length.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CmnHeader {
    /// Total command size in bytes, including this header.
    pub size: u16,
    /// Correlation tag echoed by the matching response.
    pub tag_id: u16,
    /// Message identifier (`device_ops_api_msg_e`).
    pub msg_id: u16,
    /// Command flags (see [`cmd_flags`]).
    pub flags: u16,
}

const _: () = assert!(core::mem::size_of::<CmnHeader>() == 8);

/// One DMA read transfer (`struct dma_read_node`): device DRAM to host memory.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DmaReadNode {
    /// Destination host virtual address.
    pub dst_host_virt_addr: u64,
    /// Destination host physical address (0 lets the driver resolve it).
    pub dst_host_phy_addr: u64,
    /// Source device physical address.
    pub src_device_phy_addr: u64,
    /// Transfer size in bytes.
    pub size: u32,
    pub _pad: [u8; 4],
}

const _: () = assert!(core::mem::size_of::<DmaReadNode>() == 32);

/// One DMA write transfer (`struct dma_write_node`): host memory to device DRAM.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct DmaWriteNode {
    /// Source host virtual address.
    pub src_host_virt_addr: u64,
    /// Source host physical address (0 lets the driver resolve it).
    pub src_host_phy_addr: u64,
    /// Destination device physical address.
    pub dst_device_phy_addr: u64,
    /// Transfer size in bytes.
    pub size: u32,
    pub _pad: [u8; 4],
}

const _: () = assert!(core::mem::size_of::<DmaWriteNode>() == 32);

/// U-mode trace configuration (`struct trace_init_info_t`), 40 bytes. Occupies
/// bytes `[0:40)` of a kernel-launch argument payload when tracing is enabled.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TraceInitInfo {
    /// Device address of the trace buffer.
    pub buffer: u64,
    /// Total size of the trace buffer in bytes.
    pub buffer_size: u32,
    /// Per-hart free-space threshold at which the device notifies the host.
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

const _: () = assert!(core::mem::size_of::<TraceInitInfo>() == 40);

/// U-mode stack configuration (`struct kernel_user_stack_cfg_t`), 8 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct UserStackCfg {
    /// Stack base offset from the host-managed DRAM address, in 4096-byte blocks.
    pub stack_base_offset: u32,
    /// Total stack size, in 4096-byte blocks.
    pub stack_size: u32,
}

const _: () = assert!(core::mem::size_of::<UserStackCfg>() == 8);

impl TraceInitInfo {
    /// Serialise into the 40-byte on-wire representation.
    pub fn to_bytes(&self) -> [u8; 40] {
        let mut b = [0u8; 40];
        b[0..8].copy_from_slice(&self.buffer.to_le_bytes());
        b[8..12].copy_from_slice(&self.buffer_size.to_le_bytes());
        b[12..16].copy_from_slice(&self.threshold.to_le_bytes());
        b[16..24].copy_from_slice(&self.shire_mask.to_le_bytes());
        b[24..32].copy_from_slice(&self.thread_mask.to_le_bytes());
        b[32..36].copy_from_slice(&self.event_mask.to_le_bytes());
        b[36..40].copy_from_slice(&self.filter_mask.to_le_bytes());
        b
    }
}

impl UserStackCfg {
    /// Serialise into the 8-byte on-wire representation.
    pub fn to_bytes(&self) -> [u8; 8] {
        let mut b = [0u8; 8];
        b[0..4].copy_from_slice(&self.stack_base_offset.to_le_bytes());
        b[4..8].copy_from_slice(&self.stack_size.to_le_bytes());
        b
    }
}

impl DmaReadNode {
    /// Serialise into the 32-byte on-wire representation.
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut b = [0u8; 32];
        b[0..8].copy_from_slice(&self.dst_host_virt_addr.to_le_bytes());
        b[8..16].copy_from_slice(&self.dst_host_phy_addr.to_le_bytes());
        b[16..24].copy_from_slice(&self.src_device_phy_addr.to_le_bytes());
        b[24..28].copy_from_slice(&self.size.to_le_bytes());
        b
    }
}

impl DmaWriteNode {
    /// Serialise into the 32-byte on-wire representation.
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut b = [0u8; 32];
        b[0..8].copy_from_slice(&self.src_host_virt_addr.to_le_bytes());
        b[8..16].copy_from_slice(&self.src_host_phy_addr.to_le_bytes());
        b[16..24].copy_from_slice(&self.dst_device_phy_addr.to_le_bytes());
        b[24..28].copy_from_slice(&self.size.to_le_bytes());
        b
    }
}

/// Fixed-size fields of a kernel-launch command, preceding the header's payload
/// counter but following the common header. See `device_ops_kernel_launch_cmd_t`.
const KERNEL_LAUNCH_FIXED: usize = 32; // 4 x u64

/// Build a `device_ops_kernel_launch_cmd_t` byte buffer ready for `PUSH_SQ`.
///
/// `payload` is the optional argument payload whose layout is dictated by the
/// flags: a [`TraceInitInfo`] (40 B), then a [`UserStackCfg`] (8 B), then the
/// kernel arguments, each present only when its flag is set.
pub fn build_kernel_launch(
    tag_id: u16,
    flags: u16,
    code_start_address: u64,
    pointer_to_args: u64,
    exception_buffer: u64,
    shire_mask: u64,
    payload: &[u8],
) -> Vec<u8> {
    let payload_size = KERNEL_LAUNCH_FIXED + payload.len();
    let total = CMN_HEADER_SIZE + payload_size;
    let mut buf = Vec::with_capacity(total);
    put_header(
        &mut buf,
        total as u16,
        tag_id,
        msg_id::KERNEL_LAUNCH_CMD,
        flags,
    );
    buf.extend_from_slice(&code_start_address.to_le_bytes());
    buf.extend_from_slice(&pointer_to_args.to_le_bytes());
    buf.extend_from_slice(&exception_buffer.to_le_bytes());
    buf.extend_from_slice(&shire_mask.to_le_bytes());
    buf.extend_from_slice(payload);
    buf
}

/// Build a `device_ops_dma_readlist_cmd_t` byte buffer ready for `PUSH_SQ`.
pub fn build_dma_readlist(tag_id: u16, flags: u16, nodes: &[DmaReadNode]) -> Vec<u8> {
    let total = CMN_HEADER_SIZE + core::mem::size_of_val(nodes);
    let mut buf = Vec::with_capacity(total);
    put_header(
        &mut buf,
        total as u16,
        tag_id,
        msg_id::DMA_READLIST_CMD,
        flags,
    );
    for node in nodes {
        buf.extend_from_slice(&node.to_bytes());
    }
    buf
}

/// Build a `device_ops_dma_writelist_cmd_t` byte buffer ready for `PUSH_SQ`.
pub fn build_dma_writelist(tag_id: u16, flags: u16, nodes: &[DmaWriteNode]) -> Vec<u8> {
    let total = CMN_HEADER_SIZE + core::mem::size_of_val(nodes);
    let mut buf = Vec::with_capacity(total);
    put_header(
        &mut buf,
        total as u16,
        tag_id,
        msg_id::DMA_WRITELIST_CMD,
        flags,
    );
    for node in nodes {
        buf.extend_from_slice(&node.to_bytes());
    }
    buf
}

fn put_header(buf: &mut Vec<u8>, size: u16, tag_id: u16, msg_id: u16, flags: u16) {
    buf.extend_from_slice(&size.to_le_bytes());
    buf.extend_from_slice(&tag_id.to_le_bytes());
    buf.extend_from_slice(&msg_id.to_le_bytes());
    buf.extend_from_slice(&flags.to_le_bytes());
}

/// A decoded response common header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResponseHeader {
    pub size: u16,
    pub tag_id: u16,
    pub msg_id: u16,
    pub flags: u16,
}

impl ResponseHeader {
    /// Parse the leading [`CmnHeader`] of a response buffer.
    pub fn parse(buf: &[u8]) -> Option<ResponseHeader> {
        if buf.len() < 8 {
            return None;
        }
        Some(ResponseHeader {
            size: u16::from_le_bytes([buf[0], buf[1]]),
            tag_id: u16::from_le_bytes([buf[2], buf[3]]),
            msg_id: u16::from_le_bytes([buf[4], buf[5]]),
            flags: u16::from_le_bytes([buf[6], buf[7]]),
        })
    }
}

/// Read the `status` word of a kernel-launch or DMA-list response.
///
/// Both `device_ops_kernel_launch_rsp_t` and `device_ops_dma_*list_rsp_t` share
/// the same prefix: an 8-byte response header, three 8-byte timing counters,
/// then a 32-bit status at byte offset 32.
pub const RSP_STATUS_OFFSET: usize = 8 + 8 + 8 + 8;

/// Extract the device status code from a launch/DMA response, if present.
pub fn response_status(buf: &[u8]) -> Option<u32> {
    let end = RSP_STATUS_OFFSET + 4;
    if buf.len() < end {
        return None;
    }
    Some(u32::from_le_bytes([
        buf[RSP_STATUS_OFFSET],
        buf[RSP_STATUS_OFFSET + 1],
        buf[RSP_STATUS_OFFSET + 2],
        buf[RSP_STATUS_OFFSET + 3],
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_launch_header_and_fields() {
        let payload = [0xAAu8; 40];
        let cmd = build_kernel_launch(7, cmd_flags::BARRIER, 0x8005801000, 0, 0, 0x1, &payload);

        // Common header: `size` is the total command length (header included).
        let hdr = ResponseHeader::parse(&cmd).unwrap();
        assert_eq!(
            hdr.size as usize,
            CMN_HEADER_SIZE + KERNEL_LAUNCH_FIXED + payload.len()
        );
        assert_eq!(hdr.size as usize, cmd.len());
        assert_eq!(hdr.tag_id, 7);
        assert_eq!(hdr.msg_id, msg_id::KERNEL_LAUNCH_CMD);
        assert_eq!(hdr.flags, cmd_flags::BARRIER);

        // Fixed fields begin right after the 8-byte header.
        let code_start = u64::from_le_bytes(cmd[8..16].try_into().unwrap());
        let shire_mask = u64::from_le_bytes(cmd[32..40].try_into().unwrap());
        assert_eq!(code_start, 0x8005801000);
        assert_eq!(shire_mask, 0x1);
        assert_eq!(cmd.len(), 8 + KERNEL_LAUNCH_FIXED + payload.len());
    }

    #[test]
    fn dma_readlist_layout() {
        let nodes = [
            DmaReadNode {
                dst_host_virt_addr: 0x1111,
                dst_host_phy_addr: 0,
                src_device_phy_addr: 0x80_0000_0000,
                size: 64,
                _pad: [0; 4],
            },
            DmaReadNode {
                dst_host_virt_addr: 0x2222,
                dst_host_phy_addr: 0,
                src_device_phy_addr: 0x80_0000_1000,
                size: 128,
                _pad: [0; 4],
            },
        ];
        let cmd = build_dma_readlist(3, cmd_flags::BARRIER, &nodes);
        let hdr = ResponseHeader::parse(&cmd).unwrap();
        assert_eq!(hdr.msg_id, msg_id::DMA_READLIST_CMD);
        assert_eq!(hdr.size as usize, CMN_HEADER_SIZE + nodes.len() * 32);
        assert_eq!(hdr.size as usize, cmd.len());
        // Second node's src address sits at header(8) + one node(32) + 16.
        let src1 = u64::from_le_bytes(cmd[8 + 32 + 16..8 + 32 + 24].try_into().unwrap());
        assert_eq!(src1, 0x80_0000_1000);
    }

    #[test]
    fn status_reads_at_offset_32() {
        // 8-byte header + 24 bytes of timing + status.
        let mut rsp = vec![0u8; 40];
        rsp[RSP_STATUS_OFFSET..RSP_STATUS_OFFSET + 4].copy_from_slice(&13u32.to_le_bytes());
        assert_eq!(response_status(&rsp), Some(13));
        assert_eq!(response_status(&rsp[..RSP_STATUS_OFFSET]), None);
    }
}
