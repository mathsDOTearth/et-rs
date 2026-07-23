//! Device-layer tests driven through an in-memory [`Transport`] double.
//!
//! These exercise everything up to, but not including, the kernel driver: DRAM
//! bump allocation, ELF entry recovery, kernel-launch command construction,
//! DMA read-list splitting and trace extraction. The real ioctl transport can
//! only be exercised on hardware; see `examples/hello.rs`.

use std::cell::RefCell;
use std::collections::VecDeque;

use et_soc1::proto::{self, ResponseHeader};
use et_soc1::transport::{DramInfo, PoppedResponse, Transport};
use et_soc1::{Device, LaunchOptions, Result, TraceConfig};

/// Compute-minion trace buffer type (`TRACE_BUFFER_CM`).
const TRACE_BUFFER_CM: u8 = 2;

struct MockTransport {
    dram: DramInfo,
    pushed: RefCell<Vec<(u16, Vec<u8>, u8)>>,
    responses: RefCell<VecDeque<PoppedResponse>>,
    fw_images: RefCell<Vec<Vec<u8>>>,
    cm_trace: Vec<u8>,
}

impl MockTransport {
    fn new(dram: DramInfo) -> Self {
        MockTransport {
            dram,
            pushed: RefCell::new(Vec::new()),
            responses: RefCell::new(VecDeque::new()),
            fw_images: RefCell::new(Vec::new()),
            cm_trace: Vec::new(),
        }
    }

    /// Synthesise the success response the device would return for `cmd`.
    fn canned_response(cmd: &[u8]) -> PoppedResponse {
        let hdr = ResponseHeader::parse(cmd).expect("command has a header");
        let mut rsp = vec![0u8; 40];
        // Response header: total size, echoed tag, RSP message id, no flags.
        rsp[0..2].copy_from_slice(&40u16.to_le_bytes());
        rsp[2..4].copy_from_slice(&hdr.tag_id.to_le_bytes());
        rsp[4..6].copy_from_slice(&(hdr.msg_id + 1).to_le_bytes());
        // Timing counters at offsets 8/16/24, status 0 at offset 32.
        rsp[8..16].copy_from_slice(&111u64.to_le_bytes());
        rsp[16..24].copy_from_slice(&222u64.to_le_bytes());
        rsp[24..32].copy_from_slice(&333u64.to_le_bytes());
        PoppedResponse {
            bytes: rsp,
            cq_index: 0,
        }
    }
}

impl Transport for MockTransport {
    fn dram_info(&self) -> Result<DramInfo> {
        Ok(self.dram)
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
        self.responses
            .borrow_mut()
            .push_back(Self::canned_response(cmd));
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
    // Only the launch command was pushed (load_kernel uses FW_UPDATE, not SQ).
    assert_eq!(pushed.len(), 1);
    let (sq, cmd, desc_flags) = &pushed[0];
    assert_eq!(*sq, 0);
    assert_eq!(*desc_flags, 0); // kernel launch is not a DMA descriptor

    let hdr = ResponseHeader::parse(cmd).unwrap();
    assert_eq!(hdr.msg_id, proto::msg_id::KERNEL_LAUNCH_CMD);
    let expected_flags = proto::cmd_flags::BARRIER
        | proto::cmd_flags::COMPUTE_KERNEL_TRACE
        | proto::cmd_flags::KERNEL_ARGS_EMBEDDED;
    assert_eq!(hdr.flags, expected_flags);
    // `size` is the whole command: 8-byte header + 32 fixed + 40 trace + args.
    assert_eq!(hdr.size as usize, 8 + 32 + 40 + args.len());
    assert_eq!(hdr.size as usize, cmd.len());

    // code_start_address is the ELF entry point.
    let code_start = u64::from_le_bytes(cmd[8..16].try_into().unwrap());
    assert_eq!(code_start, 0x8005801000);
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
