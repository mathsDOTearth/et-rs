//! Validates kernel loading against a real device-kernel ELF, when one has been
//! built. Exercised via the public `Device::load_kernel` path (the ELF parser is
//! crate-internal). Skips cleanly if the artifact is absent, so it does not
//! require the SDK toolchain to have run.

use std::cell::RefCell;

use et_soc1::proto::ResponseHeader;
use et_soc1::transport::{DramInfo, PoppedResponse, Transport};
use et_soc1::{Device, Result};

/// The device kernel's fixed link address (== user DRAM base on hardware).
const KERNEL_UMODE_ENTRY: u64 = 0x8005801000;

/// Captures the DMA write commands `load_kernel` issues.
struct RecordingTransport {
    dram: DramInfo,
    pushed: RefCell<Vec<Vec<u8>>>,
}

impl Transport for RecordingTransport {
    fn dram_info(&self) -> Result<DramInfo> {
        Ok(self.dram)
    }
    fn fw_update(&self, _: &[u8]) -> Result<()> {
        panic!("load_kernel must not use FW_UPDATE");
    }
    fn sq_count(&self) -> Result<u16> {
        Ok(1)
    }
    fn sq_max_msg_size(&self) -> Result<u16> {
        Ok(4096)
    }
    fn push_sq(&self, _: u16, cmd: &[u8], _: u8) -> Result<bool> {
        self.pushed.borrow_mut().push(cmd.to_vec());
        Ok(true)
    }
    fn pop_cq(&self) -> Result<Option<PoppedResponse>> {
        // Answer the just-pushed command with a success DMA response.
        let cmd = self.pushed.borrow().last().cloned();
        let Some(cmd) = cmd else {
            return Ok(None);
        };
        let hdr = ResponseHeader::parse(&cmd).unwrap();
        let mut rsp = vec![0u8; 40];
        rsp[2..4].copy_from_slice(&hdr.tag_id.to_le_bytes());
        rsp[4..6].copy_from_slice(&(hdr.msg_id + 1).to_le_bytes());
        Ok(Some(PoppedResponse {
            bytes: rsp,
            cq_index: 0,
        }))
    }
    fn extract_trace(&self, _: u8) -> Result<Vec<u8>> {
        Ok(Vec::new())
    }
}

#[test]
fn loads_real_hello_elf_when_present() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/et-testdrive/build/kernel/hello.elf"
    );
    let Ok(elf) = std::fs::read(path) else {
        eprintln!("skipping: {path} not built");
        return;
    };

    let d = Device::with_transport(RecordingTransport {
        dram: DramInfo {
            base: KERNEL_UMODE_ENTRY,
            size: 1 << 30,
            dma_max_elem_size: 0x0010_0000,
            dma_max_elem_count: 4,
            dma_alignment: 64,
        },
        pushed: RefCell::new(Vec::new()),
    })
    .unwrap();

    let kernel = d.load_kernel(&elf).unwrap();
    assert_eq!(kernel.code_start_address, KERNEL_UMODE_ENTRY);

    // The single PT_LOAD segment is DMA-written to its link address.
    let pushed = d.transport().pushed.borrow();
    assert!(!pushed.is_empty());
    let first = &pushed[0];
    let dst = u64::from_le_bytes(first[8 + 16..8 + 24].try_into().unwrap());
    assert_eq!(dst, KERNEL_UMODE_ENTRY);
}
