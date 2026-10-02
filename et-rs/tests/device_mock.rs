//! Device-layer tests driven through an in-memory [`Transport`] double.
//!
//! These exercise everything up to, but not including, the kernel driver: DRAM
//! bump allocation, ELF entry recovery, kernel-launch command construction,
//! DMA read-list splitting and trace extraction. The real ioctl transport can
//! only be exercised on hardware; see `examples/hello.rs`.

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::time::Duration;

use et_abi::ReduceArgs;
use et_soc1::proto::{self, ResponseHeader};
use et_soc1::transport::{
    DeviceConfig, DmaHostBuffer, DramInfo, PoppedResponse, Transport, VecDmaBuffer,
};
use et_soc1::{Device, DevicePod, DmaOptions, Error, LaunchOptions, Result, TraceConfig};

/// Compute-minion trace buffer type (`TRACE_BUFFER_CM`).
const TRACE_BUFFER_CM: u8 = 2;

struct MockTransport {
    dram: DramInfo,
    pushed: RefCell<Vec<(u16, Vec<u8>, u8)>>,
    responses: RefCell<VecDeque<PoppedResponse>>,
    fw_images: RefCell<Vec<Vec<u8>>>,
    cm_trace: Vec<u8>,
    /// When set, kernel-launch commands answer with this failing status and an
    /// appended `kernel_rsp_error_ptr_t` of `[exception, trace, shire_mask]`.
    launch_fail: RefCell<Option<(u32, [u64; 3])>>,
    /// When set, kernel-launch responses are withheld in `held` until
    /// `release_held` is called, modelling a completion that arrives late.
    hold_launches: Cell<bool>,
    held: RefCell<VecDeque<PoppedResponse>>,
    /// Sizes of the DMA staging buffers allocated, in order.
    staging_allocations: RefCell<Vec<usize>>,
}

impl MockTransport {
    fn new(dram: DramInfo) -> Self {
        MockTransport {
            dram,
            pushed: RefCell::new(Vec::new()),
            responses: RefCell::new(VecDeque::new()),
            fw_images: RefCell::new(Vec::new()),
            cm_trace: Vec::new(),
            launch_fail: RefCell::new(None),
            hold_launches: Cell::new(false),
            held: RefCell::new(VecDeque::new()),
            staging_allocations: RefCell::new(Vec::new()),
        }
    }

    /// Deliver every withheld launch response to the completion queue.
    fn release_held(&self) {
        let mut held = self.held.borrow_mut();
        self.responses.borrow_mut().extend(held.drain(..));
    }

    /// Build the response for a command: a configured failing launch response
    /// (echoing the command tag so `submit` correlates it), otherwise the
    /// generic success canned response.
    fn response_for(&self, cmd: &[u8]) -> PoppedResponse {
        let hdr = ResponseHeader::parse(cmd).expect("command has a header");
        if hdr.msg_id == proto::msg_id::KERNEL_LAUNCH_CMD
            && let Some((status, ptrs)) = *self.launch_fail.borrow()
        {
            let base = proto::RSP_KERNEL_ERROR_PTR_OFFSET;
            let total = (base + 24) as u16;
            let mut rsp = vec![0u8; base + 24];
            rsp[0..2].copy_from_slice(&total.to_le_bytes());
            rsp[2..4].copy_from_slice(&hdr.tag_id.to_le_bytes());
            rsp[4..6].copy_from_slice(&proto::msg_id::KERNEL_LAUNCH_RSP.to_le_bytes());
            rsp[proto::RSP_STATUS_OFFSET..proto::RSP_STATUS_OFFSET + 4]
                .copy_from_slice(&status.to_le_bytes());
            for (i, p) in ptrs.iter().enumerate() {
                let o = base + i * 8;
                rsp[o..o + 8].copy_from_slice(&p.to_le_bytes());
            }
            return PoppedResponse {
                bytes: rsp,
                cq_index: 0,
            };
        }
        Self::canned_response(cmd)
    }

    /// Synthesise the success response the device would return for `cmd`.
    ///
    /// Most responses share the kernel-launch / DMA layout: `rsp_header (8 B)
    /// + three 8-byte timing counters + status u32 (4 B) + pad (4 B) = 40 B`.
    /// The CM reset response (`device_ops_cm_reset_rsp_t`) is shorter: only
    /// `rsp_header (8 B) + status u32 (4 B) + pad (4 B) = 16 B`.
    fn canned_response(cmd: &[u8]) -> PoppedResponse {
        let hdr = ResponseHeader::parse(cmd).expect("command has a header");
        let rsp_msg_id = hdr.msg_id + 1; // CMD -> RSP is always +1 in this SDK
        let bytes = if hdr.msg_id == proto::msg_id::CM_RESET_CMD {
            // device_ops_cm_reset_rsp_t: header(8) + status(4) + pad(4) = 16 B.
            let mut rsp = vec![0u8; 16];
            rsp[0..2].copy_from_slice(&16u16.to_le_bytes());
            rsp[2..4].copy_from_slice(&hdr.tag_id.to_le_bytes());
            rsp[4..6].copy_from_slice(&rsp_msg_id.to_le_bytes());
            // status = 0 (DEV_OPS_API_CM_RESET_RESPONSE_SUCCESS) at offset 8.
            rsp
        } else {
            // Standard layout: header(8) + 3x timing(8 each) + status(4) + pad(4) = 40 B.
            let mut rsp = vec![0u8; 40];
            rsp[0..2].copy_from_slice(&40u16.to_le_bytes());
            rsp[2..4].copy_from_slice(&hdr.tag_id.to_le_bytes());
            rsp[4..6].copy_from_slice(&rsp_msg_id.to_le_bytes());
            // Timing counters at offsets 8/16/24, status 0 at offset 32.
            rsp[8..16].copy_from_slice(&111u64.to_le_bytes());
            rsp[16..24].copy_from_slice(&222u64.to_le_bytes());
            rsp[24..32].copy_from_slice(&333u64.to_le_bytes());
            rsp
        };
        PoppedResponse { bytes, cq_index: 0 }
    }
}

impl Transport for MockTransport {
    fn dram_info(&self) -> Result<DramInfo> {
        Ok(self.dram)
    }

    fn device_config(&self) -> Result<DeviceConfig> {
        // Two shires present (0 and 2), 64-byte cache line.
        Ok(DeviceConfig {
            shire_mask: 0b101,
            cache_line: 64,
        })
    }

    fn fw_update(&self, image: &[u8]) -> Result<()> {
        self.fw_images.borrow_mut().push(image.to_vec());
        Ok(())
    }

    fn sq_count(&self) -> Result<u16> {
        Ok(4)
    }

    fn sq_max_msg_size(&self) -> Result<u16> {
        Ok(4096)
    }

    fn push_sq(&self, sq_index: u16, cmd: &[u8], flags: u8) -> Result<bool> {
        let rsp = self.response_for(cmd);
        let is_launch =
            ResponseHeader::parse(cmd).unwrap().msg_id == proto::msg_id::KERNEL_LAUNCH_CMD;
        if is_launch && self.hold_launches.get() {
            self.held.borrow_mut().push_back(rsp);
        } else {
            self.responses.borrow_mut().push_back(rsp);
        }
        self.pushed
            .borrow_mut()
            .push((sq_index, cmd.to_vec(), flags));
        Ok(true)
    }

    fn pop_cq(&self) -> Result<Option<PoppedResponse>> {
        Ok(self.responses.borrow_mut().pop_front())
    }

    fn extract_trace(&self, trace_type: u8) -> Result<Vec<u8>> {
        if trace_type == TRACE_BUFFER_CM {
            Ok(self.cm_trace.clone())
        } else {
            Ok(Vec::new())
        }
    }

    fn dma_host_buffer(&self, size: usize) -> Result<Box<dyn DmaHostBuffer>> {
        self.staging_allocations.borrow_mut().push(size);
        Ok(Box::new(VecDmaBuffer::new(size)))
    }
}

/// The `(device address, size)` of every node in a DMA list command.
fn dma_nodes(cmd: &[u8]) -> Vec<(u64, u32)> {
    let size = ResponseHeader::parse(cmd).unwrap().size as usize;
    cmd[8..size]
        .as_chunks::<32>()
        .0
        .iter()
        .map(|node| {
            let addr = u64::from_le_bytes(node[16..24].try_into().unwrap());
            let len = u32::from_le_bytes(node[24..28].try_into().unwrap());
            (addr, len)
        })
        .collect()
}

fn dram(base: u64, size: u64, elem: u32, count: u16, align_bytes: u16) -> DramInfo {
    DramInfo {
        base,
        size,
        dma_max_elem_size: elem,
        dma_max_elem_count: count,
        dma_alignment: align_bytes,
    }
}

/// A minimal but valid little-endian RV64 ELF header with no program headers.
fn minimal_elf(entry: u64) -> Vec<u8> {
    let mut elf = vec![0u8; 64];
    elf[0..4].copy_from_slice(b"\x7fELF");
    elf[4] = 2; // ELFCLASS64
    elf[5] = 1; // ELFDATA2LSB
    elf[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    elf[18..20].copy_from_slice(&243u16.to_le_bytes()); // EM_RISCV
    elf[24..32].copy_from_slice(&entry.to_le_bytes()); // e_entry
    elf[54..56].copy_from_slice(&56u16.to_le_bytes()); // e_phentsize
    elf
}

/// A valid RV64 ELF with a single `PT_LOAD` segment carrying `data` at `vaddr`.
fn elf_with_segment(entry: u64, vaddr: u64, data: &[u8]) -> Vec<u8> {
    const PHOFF: usize = 64;
    const PHENT: usize = 56;
    let data_off = PHOFF + PHENT; // segment file contents follow the program header
    let mut elf = vec![0u8; data_off + data.len()];
    elf[0..4].copy_from_slice(b"\x7fELF");
    elf[4] = 2;
    elf[5] = 1;
    elf[16..18].copy_from_slice(&2u16.to_le_bytes()); // ET_EXEC
    elf[18..20].copy_from_slice(&243u16.to_le_bytes()); // EM_RISCV
    elf[24..32].copy_from_slice(&entry.to_le_bytes()); // e_entry
    elf[32..40].copy_from_slice(&(PHOFF as u64).to_le_bytes()); // e_phoff
    elf[54..56].copy_from_slice(&(PHENT as u16).to_le_bytes()); // e_phentsize
    elf[56..58].copy_from_slice(&1u16.to_le_bytes()); // e_phnum

    // One PT_LOAD program header.
    let ph = PHOFF;
    elf[ph..ph + 4].copy_from_slice(&1u32.to_le_bytes()); // p_type = PT_LOAD
    elf[ph + 8..ph + 16].copy_from_slice(&(data_off as u64).to_le_bytes()); // p_offset
    elf[ph + 16..ph + 24].copy_from_slice(&vaddr.to_le_bytes()); // p_vaddr
    elf[ph + 24..ph + 32].copy_from_slice(&vaddr.to_le_bytes()); // p_paddr
    elf[ph + 32..ph + 40].copy_from_slice(&(data.len() as u64).to_le_bytes()); // p_filesz
    elf[ph + 40..ph + 48].copy_from_slice(&(data.len() as u64).to_le_bytes()); // p_memsz

    elf[data_off..].copy_from_slice(data);
    elf
}

#[test]
fn bump_allocator_aligns_and_bounds() {
    let d = Device::with_transport(MockTransport::new(dram(
        0x80_0000_0000,
        8192,
        0x1000,
        4,
        4096,
    )))
    .unwrap();

    let a = d.alloc(100).unwrap();
    assert_eq!(a.addr, 0x80_0000_0000);
    assert_eq!(a.size, 100);

    // Next allocation is rounded up to the 4096-byte alignment.
    let b = d.alloc(8).unwrap();
    assert_eq!(b.addr, 0x80_0000_1000);

    // Only 8192 bytes exist; a further page-crossing allocation is refused.
    assert!(d.alloc(1).is_err());
}

#[test]
fn load_kernel_dma_writes_segment_and_reserves_dram() {
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 4, 4096))).unwrap();

    let code = vec![0xEEu8; 200];
    let elf = elf_with_segment(base, base, &code);
    let kernel = d.load_kernel(&elf).unwrap();
    assert_eq!(kernel.code_start_address, base);

    // The kernel is placed by a DMA write-list command, not FW_UPDATE.
    assert!(d.transport().fw_images.borrow().is_empty());
    let pushed = d.transport().pushed.borrow();
    assert_eq!(pushed.len(), 1);
    let (_, cmd, desc_flags) = &pushed[0];
    assert_eq!(*desc_flags & proto::desc_flags::DMA, proto::desc_flags::DMA);
    let hdr = ResponseHeader::parse(cmd).unwrap();
    assert_eq!(hdr.msg_id, proto::msg_id::DMA_WRITELIST_CMD);
    // Single write node: destination is the segment vaddr, size is the code length.
    let dst = u64::from_le_bytes(cmd[8 + 16..8 + 24].try_into().unwrap());
    let size = u32::from_le_bytes(cmd[8 + 24..8 + 28].try_into().unwrap());
    assert_eq!(dst, base);
    assert_eq!(size, code.len() as u32);
    drop(pushed);

    // The occupied DRAM (200 bytes, rounded up to the 4096-byte alignment) is
    // reserved, so the next allocation starts on the following page.
    let region = d.alloc(16).unwrap();
    assert_eq!(region.addr, base + 0x1000);
}

#[test]
fn launch_sets_flags_and_payload() {
    let d = Device::with_transport(MockTransport::new(dram(
        0x80_0000_0000,
        1 << 24,
        0x10000,
        4,
        4096,
    )))
    .unwrap();
    let kernel = d.load_kernel(&minimal_elf(0x8005801000)).unwrap();
    let trace_buf = d.alloc(4096).unwrap();
    let args = vec![0x5Au8; 64];
    let opts = LaunchOptions::new(0x1)
        .with_trace(TraceConfig::full(trace_buf, 0x1))
        .with_args(args.clone());

    let result = d.launch(&kernel, &opts).unwrap();
    assert_eq!(result.timing.execute_dur, 222);

    let pushed = d.transport().pushed.borrow();
    // Two commands: a DMA write staging the args into device memory, then the
    // kernel launch (load_kernel used a segment-free ELF, so it pushed nothing).
    assert_eq!(pushed.len(), 2);
    let (_, args_cmd, args_flags) = &pushed[0];
    assert_eq!(
        ResponseHeader::parse(args_cmd).unwrap().msg_id,
        proto::msg_id::DMA_WRITELIST_CMD
    );
    assert_eq!(*args_flags & proto::desc_flags::DMA, proto::desc_flags::DMA);

    let (sq, cmd, desc_flags) = &pushed[1];
    assert_eq!(*sq, 0);
    assert_eq!(*desc_flags, 0); // kernel launch is not a DMA descriptor

    let hdr = ResponseHeader::parse(cmd).unwrap();
    assert_eq!(hdr.msg_id, proto::msg_id::KERNEL_LAUNCH_CMD);
    // Args are delivered by pointer now, so the embedded flag is not set.
    let expected_flags = proto::cmd_flags::BARRIER | proto::cmd_flags::COMPUTE_KERNEL_TRACE;
    assert_eq!(hdr.flags, expected_flags);
    // `size` is the whole command: 8-byte header + 32 fixed fields + 40-byte trace.
    assert_eq!(hdr.size as usize, 8 + 32 + 40);
    assert_eq!(hdr.size as usize, cmd.len());

    // code_start_address (bytes 8..16) and a non-zero pointer_to_args (16..24).
    let code_start = u64::from_le_bytes(cmd[8..16].try_into().unwrap());
    assert_eq!(code_start, 0x8005801000);
    let pointer_to_args = u64::from_le_bytes(cmd[16..24].try_into().unwrap());
    assert_ne!(pointer_to_args, 0);
}

#[test]
fn memcpy_d2h_splits_by_dma_limits() {
    // Element size 16, at most 2 nodes per command.
    let d = Device::with_transport(MockTransport::new(dram(0x80_0000_0000, 1 << 20, 16, 2, 64)))
        .unwrap();

    let mut dst = vec![0u8; 40];
    d.memcpy_d2h(0x80_0000_0000, &mut dst).unwrap();

    let pushed = d.transport().pushed.borrow();
    // 40 bytes / 16 = 3 nodes; 2 nodes per command => 2 commands.
    assert_eq!(pushed.len(), 2);

    // Every pushed command is flagged as a DMA descriptor.
    for (_, cmd, desc_flags) in pushed.iter() {
        assert_eq!(*desc_flags & proto::desc_flags::DMA, proto::desc_flags::DMA);
        let hdr = ResponseHeader::parse(cmd).unwrap();
        assert_eq!(hdr.msg_id, proto::msg_id::DMA_READLIST_CMD);
    }

    // First command carries two 16-byte nodes (header + 2 * 32).
    let first = &pushed[0].1;
    assert_eq!(
        ResponseHeader::parse(first).unwrap().size as usize,
        8 + 2 * 32
    );
    let node0_size = u32::from_le_bytes(first[8 + 24..8 + 28].try_into().unwrap());
    assert_eq!(node0_size, 16);

    // Second command carries the remaining 8-byte node (header + 32).
    let second = &pushed[1].1;
    assert_eq!(ResponseHeader::parse(second).unwrap().size as usize, 8 + 32);
    let node_last_size = u32::from_le_bytes(second[8 + 24..8 + 28].try_into().unwrap());
    assert_eq!(node_last_size, 8);
}

#[test]
fn extract_cm_trace_returns_buffer() {
    let mut mock = MockTransport::new(dram(0x80_0000_0000, 1 << 20, 0x1000, 4, 64));
    mock.cm_trace = vec![1, 2, 3, 4];
    let d = Device::with_transport(mock).unwrap();
    assert_eq!(d.extract_cm_trace().unwrap(), vec![1, 2, 3, 4]);
}

#[test]
fn typed_buffers_size_and_upload() {
    let base = 0x80_0000_0000u64;
    let d = Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x1000, 8, 64))).unwrap();

    // alloc_array records the element count and the exact byte size.
    let a = d.alloc_array::<u32>(10).unwrap();
    assert_eq!(a.len(), 10);
    assert_eq!(a.byte_len(), 40);
    assert_eq!(a.region().size, 40);

    // alloc_padded reserves one cache line per element regardless of `T`'s size.
    let p = d.alloc_padded::<u64>(4).unwrap();
    assert_eq!(p.len(), 4);
    assert_eq!(p.stride(), 64);
    assert_eq!(p.region().size, 256);

    // upload issues a single DMA write of exactly the slice's byte length.
    let buf = d.upload(&[1u32, 2, 3]).unwrap();
    assert_eq!(buf.byte_len(), 12);
    let pushed = d.transport().pushed.borrow();
    let (_, cmd, desc) = pushed.last().unwrap();
    assert_eq!(*desc & proto::desc_flags::DMA, proto::desc_flags::DMA);
    assert_eq!(
        ResponseHeader::parse(cmd).unwrap().msg_id,
        proto::msg_id::DMA_WRITELIST_CMD
    );
    // node0 size field lives at header (8) + node offset (24).
    let node0_size = u32::from_le_bytes(cmd[8 + 24..8 + 28].try_into().unwrap());
    assert_eq!(node0_size, 12);
}

#[test]
fn topology_from_device_config() {
    let d = Device::with_transport(MockTransport::new(dram(
        0x80_0000_0000,
        1 << 20,
        0x1000,
        4,
        64,
    )))
    .unwrap();
    let t = d.topology().unwrap();

    // Device-queried fields come from the transport's device_config...
    assert_eq!(t.shire_mask, 0b101);
    assert_eq!(t.cache_line, 64);
    // ...the per-shire geometry is architectural.
    assert_eq!(t.harts_per_shire, 64);
    assert_eq!(t.harts_per_neighbourhood, 16);

    assert_eq!(t.num_shires(), 2);
    assert_eq!(t.num_harts(), 128);
    assert_eq!(t.first_shire(), 0b001);
}

#[test]
fn alloc_mark_and_reset_reclaims() {
    let base = 0x80_0000_0000u64;
    let d = Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x1000, 4, 64))).unwrap();

    let _a = d.alloc(128).unwrap();
    let mark = d.alloc_mark();
    let b = d.alloc(256).unwrap();
    d.reset_to(mark);

    // The next allocation reclaims exactly the span released by the reset.
    let c = d.alloc(256).unwrap();
    assert_eq!(c.addr, b.addr);
}

#[test]
fn launch_reuses_args_scratch() {
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 8, 4096))).unwrap();
    let kernel = d
        .load_kernel(&elf_with_segment(base, base, &[0x13, 0, 0, 0]))
        .unwrap();
    let opts = LaunchOptions::new(0x1).with_args(vec![0u8; 24]);

    d.launch(&kernel, &opts).unwrap();
    let after_first = d.dram_available();
    d.launch(&kernel, &opts).unwrap();
    let after_second = d.dram_available();

    // The second launch reuses the argument scratch rather than leaking a region.
    assert_eq!(after_first, after_second);
}

#[test]
fn launch_spmd_sets_shire_and_stages_args() {
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 24, 0x10000, 4, 4096))).unwrap();
    let kernel = d.load_kernel(&minimal_elf(base)).unwrap();
    let args = ReduceArgs {
        input: 0x1000,
        out: 0x2000,
        n: 10,
        n_harts: 64,
    };

    d.launch_spmd(&kernel, 0b101, &args).unwrap();

    let pushed = d.transport().pushed.borrow();
    // Args staged by a DMA write, then the launch command (segment-free ELF, so
    // load_kernel pushed nothing).
    assert_eq!(pushed.len(), 2);
    assert_eq!(
        ResponseHeader::parse(&pushed[0].1).unwrap().msg_id,
        proto::msg_id::DMA_WRITELIST_CMD
    );
    let launch_cmd = &pushed[1].1;
    assert_eq!(
        ResponseHeader::parse(launch_cmd).unwrap().msg_id,
        proto::msg_id::KERNEL_LAUNCH_CMD
    );
    // The shire mask sits at bytes [32..40] of the launch command.
    let shire_mask = u64::from_le_bytes(launch_cmd[32..40].try_into().unwrap());
    assert_eq!(shire_mask, 0b101);
}

#[test]
fn launch_surfaces_exception_detail() {
    let base = 0x80_0000_0000u64;
    let mut mock = MockTransport::new(dram(base, 1 << 20, 0x10000, 4, 4096));
    // The device will fault this launch with EXCEPTION(2) and append the
    // exception/trace pointers and faulting shire mask.
    mock.launch_fail = RefCell::new(Some((2, [0xAAAA, 0xBBBB, 0x1])));
    let d = Device::with_transport(mock).unwrap();

    let kernel = d
        .load_kernel(&elf_with_segment(base, base, &[0x13, 0, 0, 0]))
        .unwrap();
    let err = d.launch(&kernel, &LaunchOptions::new(0x1)).unwrap_err();

    match err {
        Error::KernelLaunch {
            status,
            status_name,
            detail,
        } => {
            assert_eq!(status, 2);
            assert_eq!(status_name, "EXCEPTION");
            let d = detail.expect("device appended the error pointers");
            assert_eq!(d.exception_buffer, 0xAAAA);
            assert_eq!(d.trace_buffer, 0xBBBB);
            assert_eq!(d.shire_mask, 0x1);
        }
        other => panic!("expected KernelLaunch error, got {other:?}"),
    }
}

/// A transport that pushes commands (so `push_cmd` succeeds) but never
/// enqueues a CQ response (so `collect_response` must exhaust the deadline).
/// Used to exercise `Error::Timeout` without sleeping for the full default.
struct NullCqTransport {
    dram: DramInfo,
    pushed: RefCell<Vec<(u16, Vec<u8>, u8)>>,
}

impl NullCqTransport {
    fn new(dram: DramInfo) -> Self {
        NullCqTransport {
            dram,
            pushed: RefCell::new(Vec::new()),
        }
    }
}

impl Transport for NullCqTransport {
    fn dram_info(&self) -> Result<DramInfo> {
        Ok(self.dram)
    }
    fn sq_count(&self) -> Result<u16> {
        Ok(2)
    }
    fn sq_max_msg_size(&self) -> Result<u16> {
        Ok(4096)
    }
    fn push_sq(&self, sq_index: u16, cmd: &[u8], flags: u8) -> Result<bool> {
        // Record the command but do not enqueue a response.
        self.pushed
            .borrow_mut()
            .push((sq_index, cmd.to_vec(), flags));
        Ok(true)
    }
    fn pop_cq(&self) -> Result<Option<PoppedResponse>> {
        Ok(None)
    }
    fn extract_trace(&self, _: u8) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
    fn fw_update(&self, _image: &[u8]) -> Result<()> {
        unimplemented!("NullCqTransport does not support firmware update")
    }
}

#[test]
fn timeout_error_carries_operation_and_limit() {
    // `minimal_elf` has no load segments, so `load_kernel` does no DMA and the
    // NullCqTransport only needs to handle the kernel-launch push.
    let base = 0x80_0000_0000u64;
    let d = Device::with_transport(NullCqTransport::new(dram(base, 1 << 20, 0x10000, 4, 4096)))
        .unwrap();
    // Set a very short timeout so the test does not block for the default 10 s.
    d.set_default_launch_timeout(Duration::from_millis(10));
    let kernel = d.load_kernel(&minimal_elf(base)).unwrap();

    let err = d.launch(&kernel, &LaunchOptions::new(0x1)).unwrap_err();
    match err {
        Error::Timeout { operation, limit } => {
            assert_eq!(operation, "kernel completion");
            assert_eq!(limit, Duration::from_millis(10));
        }
        other => panic!("expected Error::Timeout, got {other:?}"),
    }
}

#[test]
fn per_launch_timeout_takes_precedence_over_device_default() {
    let base = 0x80_0000_0000u64;
    let d = Device::with_transport(NullCqTransport::new(dram(base, 1 << 20, 0x10000, 4, 4096)))
        .unwrap();
    // Device default is long; per-launch override is short.
    d.set_default_launch_timeout(Duration::from_secs(3600));
    let kernel = d.load_kernel(&minimal_elf(base)).unwrap();
    let opts = LaunchOptions::new(0x1).with_timeout(Duration::from_millis(10));

    let err = d.launch(&kernel, &opts).unwrap_err();
    match err {
        Error::Timeout { limit, .. } => {
            assert_eq!(limit, Duration::from_millis(10));
        }
        other => panic!("expected Error::Timeout, got {other:?}"),
    }
}

#[test]
fn launch_spmd_opts_dispatches_with_opts_shire_mask() {
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 24, 0x10000, 4, 4096))).unwrap();
    let kernel = d.load_kernel(&minimal_elf(base)).unwrap();
    let args = ReduceArgs {
        input: 0x1000,
        out: 0x2000,
        n: 8,
        n_harts: 32,
    };
    let opts = LaunchOptions::new(0b110).with_timeout(Duration::from_secs(60));

    d.launch_spmd_opts(&kernel, &args, opts).unwrap();

    let pushed = d.transport().pushed.borrow();
    // DMA write for args staging, then the kernel-launch command.
    assert_eq!(pushed.len(), 2);
    let launch_cmd = &pushed[1].1;
    assert_eq!(
        ResponseHeader::parse(launch_cmd).unwrap().msg_id,
        proto::msg_id::KERNEL_LAUNCH_CMD
    );
    let shire_mask = u64::from_le_bytes(launch_cmd[32..40].try_into().unwrap());
    assert_eq!(shire_mask, 0b110, "shire_mask must come from the opts");
}

#[test]
fn dma_options_with_timeout_issues_command() {
    // Verifies that DmaOptions::with_timeout does not disrupt the DMA path.
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 4, 4096))).unwrap();
    let opts = DmaOptions::new()
        .on_sq(1)
        .with_timeout(Duration::from_secs(30));
    let mut dst = vec![0u8; 64];

    d.memcpy_d2h_opts(base, &mut dst, &opts).unwrap();

    let pushed = d.transport().pushed.borrow();
    assert_eq!(pushed.len(), 1, "one DMA read-list command expected");
    assert_eq!(
        ResponseHeader::parse(&pushed[0].1).unwrap().msg_id,
        proto::msg_id::DMA_READLIST_CMD
    );
    // Command must be on SQ 1.
    assert_eq!(pushed[0].0, 1);
}

#[test]
fn upload_slice_issues_writelist_of_correct_byte_length() {
    // upload_slice<f32> of 4 elements = 16 bytes. Verifies the DMA write-list
    // command is issued with a single node covering exactly 16 bytes.
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 4, 4096))).unwrap();
    let data = [1.0f32, 2.0, 3.0, 4.0];

    d.upload_slice(&data, base).unwrap();

    let pushed = d.transport().pushed.borrow();
    assert_eq!(pushed.len(), 1, "one DMA write-list command expected");
    let cmd = &pushed[0].1;
    assert_eq!(
        ResponseHeader::parse(cmd).unwrap().msg_id,
        proto::msg_id::DMA_WRITELIST_CMD
    );
    // Node size field: header (8) + node offset (24).
    let node_size = u32::from_le_bytes(cmd[8 + 24..8 + 28].try_into().unwrap());
    assert_eq!(node_size, 16, "node must cover 4 * 4 = 16 bytes");
}

#[test]
fn upload_slice_opts_routes_to_specified_sq() {
    // upload_slice_opts with on_sq(1) must issue the DMA command on SQ 1.
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 4, 4096))).unwrap();
    let data = [0u32; 8];
    let opts = DmaOptions::new().on_sq(1);

    d.upload_slice_opts(&data, base, &opts).unwrap();

    let pushed = d.transport().pushed.borrow();
    assert_eq!(pushed.len(), 1);
    assert_eq!(pushed[0].0, 1, "upload_slice_opts must use the supplied SQ");
}

#[test]
fn device_pod_impl_for_external_repr_c_struct() {
    // Confirms that a user-defined #[repr(C)] struct in a separate module can
    // impl DevicePod (the orphan rule is satisfied because DevicePod now lives
    // in et_abi, not et_soc1). The test is a compile-time check: if DevicePod
    // were still in et_soc1 this block would not compile with the impl below.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct MyVertex {
        x: f32,
        y: f32,
        z: f32,
    }
    // SAFETY: MyVertex is #[repr(C)], all-float, no padding, valid for any bits.
    unsafe impl DevicePod for MyVertex {}

    // Ensure the trait is usable via the et_abi path as well.
    fn accepts_pod<T: et_abi::DevicePod>(_: &T) {}
    accepts_pod(&MyVertex {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    });
}

#[test]
fn reset_shires_issues_cm_reset_cmd() {
    // Verifies that reset_shires submits a CM_RESET_CMD with the supplied
    // shire mask, then returns Ok(()) on a success response.
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 4, 4096))).unwrap();

    d.reset_shires(0b1010).unwrap();

    let pushed = d.transport().pushed.borrow();
    assert_eq!(pushed.len(), 1, "one command expected");
    let (sq, cmd, desc) = &pushed[0];
    assert_eq!(*sq, 0);
    // CMD_DESC_FLAG_MM_RESET is NOT valid for user-space PUSH_SQ (driver
    // returns EINVAL). CM reset uses the standard descriptor flags (0).
    // CMD_DESC_FLAG_HIGH_PRIORITY routes the command to the HPSQ, which is
    // the MM firmware path that handles CM reset on real hardware.
    assert_eq!(
        *desc & proto::desc_flags::HIGH_PRIORITY,
        proto::desc_flags::HIGH_PRIORITY,
        "CM reset must carry the HIGH_PRIORITY descriptor flag (HPSQ path)"
    );
    let hdr = ResponseHeader::parse(cmd).unwrap();
    assert_eq!(hdr.msg_id, proto::msg_id::CM_RESET_CMD);
    // BARRIER flag must be set in the command header to serialise against prior
    // kernel launches.
    assert_eq!(
        hdr.flags & proto::cmd_flags::BARRIER,
        proto::cmd_flags::BARRIER,
        "CM reset command must carry the BARRIER cmd flag"
    );
    // shire_mask is the 8 bytes immediately following the 8-byte header.
    let mask = u64::from_le_bytes(cmd[8..16].try_into().unwrap());
    assert_eq!(mask, 0b1010, "shire_mask must be forwarded verbatim");
}

/// Commands of the given message type, in submission order.
fn pushed_of(d: &Device<MockTransport>, msg_id: u16) -> Vec<Vec<u8>> {
    d.transport()
        .pushed
        .borrow()
        .iter()
        .filter(|(_, cmd, _)| ResponseHeader::parse(cmd).unwrap().msg_id == msg_id)
        .map(|(_, cmd, _)| cmd.clone())
        .collect()
}

/// The `pointer_to_args` field (bytes [16..24]) of a kernel-launch command.
fn launch_args_pointer(cmd: &[u8]) -> u64 {
    u64::from_le_bytes(cmd[16..24].try_into().unwrap())
}

#[test]
fn abandoned_tag_is_not_reissued_until_its_late_response_arrives() {
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 4, 4096))).unwrap();
    let kernel = d.load_kernel(&minimal_elf(base)).unwrap();

    // The first launch times out; its (failing) completion is withheld.
    d.transport().hold_launches.set(true);
    *d.transport().launch_fail.borrow_mut() = Some((1, [0; 3]));
    let opts = LaunchOptions::new(0x1).with_timeout(Duration::from_millis(10));
    assert!(matches!(
        d.launch(&kernel, &opts).unwrap_err(),
        Error::Timeout { .. }
    ));
    let abandoned_tag = ResponseHeader::parse(&pushed_of(&d, proto::msg_id::KERNEL_LAUNCH_CMD)[0])
        .unwrap()
        .tag_id;
    d.transport().hold_launches.set(false);
    *d.transport().launch_fail.borrow_mut() = None;

    // Cycle through the whole 16-bit tag space. Every launch must succeed and
    // none may reuse the abandoned tag.
    for _ in 0..=u16::MAX as usize {
        d.launch(&kernel, &LaunchOptions::new(0x1)).unwrap();
    }
    let launches = pushed_of(&d, proto::msg_id::KERNEL_LAUNCH_CMD);
    assert!(
        launches[1..]
            .iter()
            .all(|cmd| ResponseHeader::parse(cmd).unwrap().tag_id != abandoned_tag)
    );

    // The late failing completion now arrives. It must be discarded rather
    // than reported as the result of the next launch.
    d.transport().release_held();
    d.launch(&kernel, &LaunchOptions::new(0x1)).unwrap();
    assert!(d.transport().responses.borrow().is_empty());
}

#[test]
fn args_slot_is_not_reused_while_its_launch_is_outstanding() {
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 8, 4096))).unwrap();
    let kernel = d.load_kernel(&minimal_elf(base)).unwrap();
    let opts = LaunchOptions::new(0x1).with_args(vec![7u8; 24]);

    let first = d.launch_async(&kernel, &opts).unwrap();
    let second = d.launch_async(&kernel, &opts).unwrap();
    let launches = pushed_of(&d, proto::msg_id::KERNEL_LAUNCH_CMD);
    let (p1, p2) = (
        launch_args_pointer(&launches[0]),
        launch_args_pointer(&launches[1]),
    );
    assert_ne!(
        p1, p2,
        "a running launch's arguments must not be overwritten"
    );

    d.wait_launch(first).unwrap();
    d.wait_launch(second).unwrap();

    // Both launches are collected, so a third reuses an existing slot.
    let available = d.dram_available();
    d.launch(&kernel, &opts).unwrap();
    let p3 = launch_args_pointer(
        pushed_of(&d, proto::msg_id::KERNEL_LAUNCH_CMD)
            .last()
            .unwrap(),
    );
    assert!(p3 == p1 || p3 == p2);
    assert_eq!(d.dram_available(), available);
}

#[test]
fn args_dma_is_collected_by_wait_launch_on_the_launch_queue() {
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 8, 4096))).unwrap();
    let kernel = d.load_kernel(&minimal_elf(base)).unwrap();
    let opts = LaunchOptions::new(0x1).on_sq(1).with_args(vec![3u8; 24]);

    // With BARRIER the argument DMA is pushed but not awaited: both responses
    // are still queued when `launch_async` returns.
    let pending = d.launch_async(&kernel, &opts).unwrap();
    assert_eq!(d.transport().responses.borrow().len(), 2);
    {
        let pushed = d.transport().pushed.borrow();
        let (args_sq, args_cmd, _) = &pushed[0];
        assert_eq!(
            ResponseHeader::parse(args_cmd).unwrap().msg_id,
            proto::msg_id::DMA_WRITELIST_CMD
        );
        assert_eq!(*args_sq, 1, "argument DMA must share the launch's queue");
        assert_eq!(pushed[1].0, 1);
    }
    d.wait_launch(pending).unwrap();
    assert!(d.transport().responses.borrow().is_empty());

    // Without BARRIER the argument DMA is awaited before the launch is pushed.
    let unbarriered = opts.clone().without_barrier();
    let pending = d.launch_async(&kernel, &unbarriered).unwrap();
    assert_eq!(d.transport().responses.borrow().len(), 1);
    d.wait_launch(pending).unwrap();
    assert!(d.transport().responses.borrow().is_empty());
}

#[test]
fn args_host_buffer_is_mapped_once_per_slot() {
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 8, 4096))).unwrap();
    let kernel = d.load_kernel(&minimal_elf(base)).unwrap();
    let opts = LaunchOptions::new(0x1).with_args(vec![9u8; 24]);

    for _ in 0..3 {
        d.launch(&kernel, &opts).unwrap();
    }
    // One host buffer for the single slot; the general staging buffers are
    // not used by argument staging.
    assert_eq!(d.transport().staging_allocations.borrow().len(), 1);

    // Two concurrent launches need a second slot, and so a second buffer.
    let first = d.launch_async(&kernel, &opts).unwrap();
    let second = d.launch_async(&kernel, &opts).unwrap();
    d.wait_launch(first).unwrap();
    d.wait_launch(second).unwrap();
    assert_eq!(d.transport().staging_allocations.borrow().len(), 2);
}

#[test]
fn load_kernel_rejects_overlap_with_live_allocations() {
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 4, 4096))).unwrap();
    let _buffer = d.alloc(4096).unwrap();

    let err = d
        .load_kernel(&elf_with_segment(base, base, &[0x13, 0, 0, 0]))
        .unwrap_err();
    assert!(matches!(err, Error::Limit(ref msg) if msg.contains("overlaps live allocations")));
    // Validation precedes any DMA, so nothing was written.
    assert!(d.transport().pushed.borrow().is_empty());
}

#[test]
fn reset_to_never_reclaims_kernel_image() {
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 4, 4096))).unwrap();
    let mark = d.alloc_mark();
    d.load_kernel(&elf_with_segment(base, base, &[0x13, 0, 0, 0]))
        .unwrap();

    d.reset_to(mark);
    let region = d.alloc(64).unwrap();
    assert!(
        region.addr >= base + 4,
        "allocation overlaps the kernel image"
    );
}

#[test]
fn sgemm_rejects_short_stride_and_absent_shires() {
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 4, 4096))).unwrap();
    let kernel = d.load_kernel(&minimal_elf(base)).unwrap();
    let (a, b, c) = (base + 0x1000, base + 0x2000, base + 0x3000);

    // K = 32 f32 values need 128 bytes per row of A; a 64-byte stride is short.
    let err =
        et_soc1::sgemm(&d, &kernel, 16, 16, 32, 1.0, a, 64, b, 64, 0.0, c, 64, 1).unwrap_err();
    assert!(matches!(err, Error::Limit(ref msg) if msg.contains("shorter than one row")));

    // The mock topology provides shires 0 and 2 only; n_shires = 2 needs 0 and 1.
    let err =
        et_soc1::sgemm(&d, &kernel, 16, 16, 16, 1.0, a, 64, b, 64, 0.0, c, 64, 2).unwrap_err();
    assert!(matches!(err, Error::Limit(ref msg) if msg.contains("requires shires")));

    // Neither rejected call reached the device.
    assert!(pushed_of(&d, proto::msg_id::KERNEL_LAUNCH_CMD).is_empty());
}

#[test]
fn staging_buffer_is_reused_and_grown_by_powers_of_two() {
    let d = Device::with_transport(MockTransport::new(dram(
        0x80_0000_0000,
        1 << 20,
        0x1000,
        8,
        64,
    )))
    .unwrap();
    let mut dst = vec![0u8; 100];
    d.memcpy_d2h(0x80_0000_0000, &mut dst).unwrap();
    d.memcpy_h2d(&dst, 0x80_0000_0000).unwrap();
    d.memcpy_h2d(&dst[..10], 0x80_0000_0000).unwrap();
    // Transfers that fit the retained buffer allocate nothing further.
    assert_eq!(*d.transport().staging_allocations.borrow(), vec![128]);

    // A larger transfer replaces it with the next power of two.
    d.memcpy_h2d(&[0u8; 300], 0x80_0000_0000).unwrap();
    assert_eq!(*d.transport().staging_allocations.borrow(), vec![128, 512]);
}

#[test]
fn transfers_above_half_staging_capacity_are_pipelined_with_barriers() {
    let base = 0x80_0000_0000u64;
    let d = Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x1000, 8, 64))).unwrap();
    // Rounded down to a multiple of twice the 64-byte alignment.
    d.set_staging_capacity(300);
    assert_eq!(d.staging_capacity(), 256);

    let mut dst = vec![0u8; 300];
    d.memcpy_d2h(base, &mut dst).unwrap();
    d.fill(
        et_soc1::DeviceRegion {
            addr: base,
            size: 300,
        },
        0xFF,
    )
    .unwrap();

    // Two half-capacity buffers, allocated once and shared by both transfers.
    assert_eq!(*d.transport().staging_allocations.borrow(), vec![128, 128]);
    let pushed = d.transport().pushed.borrow();
    let expected = [(base, 128), (base + 128, 128), (base + 256, 44)];
    assert_eq!(pushed.len(), 2 * expected.len());
    for (index, (_, cmd, _)) in pushed.iter().enumerate() {
        let hdr = ResponseHeader::parse(cmd).unwrap();
        let wanted_msg = if index < expected.len() {
            proto::msg_id::DMA_READLIST_CMD
        } else {
            proto::msg_id::DMA_WRITELIST_CMD
        };
        assert_eq!(hdr.msg_id, wanted_msg);
        assert_eq!(
            hdr.flags & proto::cmd_flags::BARRIER,
            proto::cmd_flags::BARRIER
        );
        assert_eq!(dma_nodes(cmd), vec![expected[index % expected.len()]]);
    }

    // A transfer within half the capacity needs only the first buffer.
    drop(pushed);
    d.memcpy_h2d(&[0u8; 100], base).unwrap();
    assert_eq!(d.transport().staging_allocations.borrow().len(), 2);
}

/// A staging buffer whose bytes may be written through its virtual address by
/// [`MemoryTransport`] in the manner of a device DMA engine, outside any Rust
/// borrow of the buffer.
struct CellDmaBuffer {
    bytes: Box<[std::cell::UnsafeCell<u8>]>,
}

impl DmaHostBuffer for CellDmaBuffer {
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: `UnsafeCell<u8>` has the layout of `u8`, and `&mut self`
        // excludes every other access for the lifetime of the slice.
        unsafe { std::slice::from_raw_parts_mut(self.bytes.as_ptr() as *mut u8, self.bytes.len()) }
    }
    fn as_slice(&self) -> &[u8] {
        // SAFETY: as above; the transport writes only within `push_sq`, during
        // which the device holds no slice of the buffer.
        unsafe { std::slice::from_raw_parts(self.bytes.as_ptr() as *const u8, self.bytes.len()) }
    }
    fn virt_addr(&self) -> u64 {
        self.bytes.as_ptr() as u64
    }
    fn phys_addr(&self) -> u64 {
        0
    }
}

/// A transport that executes DMA list commands against an in-memory model of
/// device DRAM, so that the data path of `memcpy_h2d`, `memcpy_d2h` and `fill`
/// is checked byte for byte.
struct MemoryTransport {
    dram: DramInfo,
    memory: RefCell<Vec<u8>>,
    responses: RefCell<VecDeque<PoppedResponse>>,
}

impl MemoryTransport {
    fn new(dram: DramInfo) -> Self {
        MemoryTransport {
            dram,
            memory: RefCell::new(vec![0u8; dram.size as usize]),
            responses: RefCell::new(VecDeque::new()),
        }
    }
}

impl Transport for MemoryTransport {
    fn dram_info(&self) -> Result<DramInfo> {
        Ok(self.dram)
    }
    fn fw_update(&self, _image: &[u8]) -> Result<()> {
        Ok(())
    }
    fn sq_count(&self) -> Result<u16> {
        Ok(1)
    }
    fn sq_max_msg_size(&self) -> Result<u16> {
        Ok(4096)
    }
    fn push_sq(&self, _sq_index: u16, cmd: &[u8], _flags: u8) -> Result<bool> {
        let hdr = ResponseHeader::parse(cmd).unwrap();
        let size = hdr.size as usize;
        let mut memory = self.memory.borrow_mut();
        for node in cmd[8..size].as_chunks::<32>().0 {
            let host = u64::from_le_bytes(node[0..8].try_into().unwrap()) as *mut u8;
            let device = u64::from_le_bytes(node[16..24].try_into().unwrap());
            let len = u32::from_le_bytes(node[24..28].try_into().unwrap()) as usize;
            let start = (device - self.dram.base) as usize;
            let region = &mut memory[start..start + len];
            // SAFETY: `host` addresses a live `CellDmaBuffer` of at least `len`
            // bytes, which the device does not borrow during `push_sq`.
            unsafe {
                if hdr.msg_id == proto::msg_id::DMA_WRITELIST_CMD {
                    std::ptr::copy_nonoverlapping(host, region.as_mut_ptr(), len);
                } else {
                    assert_eq!(hdr.msg_id, proto::msg_id::DMA_READLIST_CMD);
                    std::ptr::copy_nonoverlapping(region.as_ptr(), host, len);
                }
            }
        }
        self.responses
            .borrow_mut()
            .push_back(MockTransport::canned_response(cmd));
        Ok(true)
    }
    fn pop_cq(&self) -> Result<Option<PoppedResponse>> {
        Ok(self.responses.borrow_mut().pop_front())
    }
    fn extract_trace(&self, _trace_type: u8) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
    fn dma_host_buffer(&self, size: usize) -> Result<Box<dyn DmaHostBuffer>> {
        let bytes = (0..size).map(|_| std::cell::UnsafeCell::new(0)).collect();
        Ok(Box::new(CellDmaBuffer { bytes }))
    }
}

#[test]
fn pipelined_transfers_and_fill_preserve_data() {
    let base = 0x80_0000_0000u64;
    // Element size 96 forces several nodes per chunk and, with two nodes per
    // command, several commands per chunk.
    let d = Device::with_transport(MemoryTransport::new(dram(base, 1 << 16, 96, 2, 64))).unwrap();
    d.set_staging_capacity(512);

    for total in [1usize, 63, 256, 257, 700, 1000, 4096 + 13] {
        let pattern: Vec<u8> = (0..total).map(|i| (i * 7 + total) as u8).collect();
        d.memcpy_h2d(&pattern, base).unwrap();
        let mut readback = vec![0u8; total];
        d.memcpy_d2h(base, &mut readback).unwrap();
        assert_eq!(readback, pattern, "round trip of {total} bytes");

        let value = total as u8 ^ 0xA5;
        d.fill(
            et_soc1::DeviceRegion {
                addr: base,
                size: total as u64,
            },
            value,
        )
        .unwrap();
        d.memcpy_d2h(base, &mut readback).unwrap();
        assert!(
            readback.iter().all(|&b| b == value),
            "fill of {total} bytes"
        );
    }
}

#[test]
fn pinned_transfers_preserve_data_without_staging() {
    let base = 0x80_0000_0000u64;
    let d = Device::with_transport(MemoryTransport::new(dram(base, 1 << 16, 96, 2, 64))).unwrap();
    d.set_staging_capacity(512);

    for total in [1usize, 63, 700, 4096 + 13] {
        let mut outbound = d.alloc_pinned(total).unwrap();
        assert_eq!(outbound.len(), total);
        for (i, byte) in outbound.as_mut_slice().iter_mut().enumerate() {
            *byte = (i * 11 + total) as u8;
        }
        d.memcpy_h2d_pinned(&mut outbound, base).unwrap();

        let mut inbound = d.alloc_pinned(total).unwrap();
        d.memcpy_d2h_pinned(base, &mut inbound).unwrap();
        assert_eq!(inbound.as_slice(), outbound.as_slice(), "{total} bytes");

        // Interoperates with the staged path.
        let mut readback = vec![0u8; total];
        d.memcpy_d2h(base, &mut readback).unwrap();
        assert_eq!(readback, outbound.as_slice());
    }
}

#[test]
fn pinned_transfer_issues_no_staging_and_rejects_foreign_buffers() {
    let base = 0x80_0000_0000u64;
    let d =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 8, 4096))).unwrap();
    let mut pinned = d.alloc_pinned(256).unwrap();
    d.memcpy_h2d_pinned(&mut pinned, base).unwrap();
    d.memcpy_d2h_pinned(base, &mut pinned).unwrap();
    // Only the pinned buffer itself was mapped.
    assert_eq!(*d.transport().staging_allocations.borrow(), [256]);
    {
        let pushed = d.transport().pushed.borrow();
        assert_eq!(pushed.len(), 2);
        assert_eq!(dma_nodes(&pushed[0].1), [(base, 256)]);
        assert_eq!(dma_nodes(&pushed[1].1), [(base, 256)]);
    }

    let other =
        Device::with_transport(MockTransport::new(dram(base, 1 << 20, 0x10000, 8, 4096))).unwrap();
    let err = other.memcpy_h2d_pinned(&mut pinned, base).unwrap_err();
    assert!(matches!(err, Error::Limit(ref msg) if msg.contains("different device")));
    assert!(other.transport().pushed.borrow().is_empty());
}
