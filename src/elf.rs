//! Minimal ELF64 inspection, sufficient to load a device kernel.
//!
//! Device kernels are little-endian RV64 ELF executables linked at a fixed
//! U-mode address that coincides with the base of the user DRAM region. Loading
//! one means copying its `PT_LOAD` segments to their virtual addresses in device
//! DRAM (over DMA) and then launching at `e_entry`; there is no firmware-side
//! ELF loader for compute kernels. This module extracts just what that requires:
//! the entry point and the loadable segments.

use crate::error::{Error, Result};

/// `EM_RISCV`, the ELF machine identifier for RISC-V.
const EM_RISCV: u16 = 243;
/// `PT_LOAD` segment type.
const PT_LOAD: u32 = 1;
/// Size of an ELF64 program-header entry.
const PHENT_SIZE: usize = 56;

/// One loadable (`PT_LOAD`) segment of a device kernel image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadSegment {
    /// Byte offset of the segment's file contents within the image.
    pub file_offset: u64,
    /// Device virtual (== physical) address the segment loads to.
    pub vaddr: u64,
    /// Number of bytes present in the file for this segment.
    pub file_size: u64,
    /// Number of bytes the segment occupies in memory (`>= file_size`; the
    /// excess is zero-initialised `.bss`).
    pub mem_size: u64,
}

/// A parsed device-kernel ELF image: its entry point and loadable segments.
#[derive(Clone, Debug)]
pub struct KernelImage {
    /// Entry-point virtual address (`e_entry`), used as the launch
    /// `code_start_address`.
    pub entry: u64,
    /// Loadable segments, in program-header order.
    pub segments: Vec<LoadSegment>,
}

/// Parse a 64-bit little-endian RISC-V ELF image.
pub fn parse(image: &[u8]) -> Result<KernelImage> {
    // ELF64 header is 64 bytes.
    if image.len() < 64 {
        return Err(Error::Elf("image shorter than an ELF64 header".into()));
    }
    if &image[0..4] != b"\x7fELF" {
        return Err(Error::Elf("bad ELF magic".into()));
    }
    if image[4] != 2 {
        return Err(Error::Elf("not an ELF64 image".into()));
    }
    if image[5] != 1 {
        return Err(Error::Elf("not a little-endian image".into()));
    }
    let machine = u16::from_le_bytes([image[18], image[19]]);
    if machine != EM_RISCV {
        return Err(Error::Elf(format!(
            "unexpected machine {machine} (expected RISC-V {EM_RISCV})"
        )));
    }

    let entry = rd_u64(image, 24)?;
    let phoff = rd_u64(image, 32)? as usize;
    let phentsize = rd_u16(image, 54)? as usize;
    let phnum = rd_u16(image, 56)? as usize;

    if phentsize != PHENT_SIZE {
        return Err(Error::Elf(format!(
            "unexpected program-header entry size {phentsize}"
        )));
    }

    let mut segments = Vec::new();
    for i in 0..phnum {
        let base = phoff + i * PHENT_SIZE;
        if base + PHENT_SIZE > image.len() {
            return Err(Error::Elf("program header table out of bounds".into()));
        }
        // ELF64 program header field offsets.
        let p_type = rd_u32(image, base)?;
        if p_type != PT_LOAD {
            continue;
        }
        let file_offset = rd_u64(image, base + 8)?;
        let vaddr = rd_u64(image, base + 16)?;
        let file_size = rd_u64(image, base + 32)?;
        let mem_size = rd_u64(image, base + 40)?;

        // Validate the segment's file contents lie within the image.
        let end = file_offset
            .checked_add(file_size)
            .ok_or_else(|| Error::Elf("segment file range overflows".into()))?;
        if end > image.len() as u64 {
            return Err(Error::Elf("segment file range out of bounds".into()));
        }
        if mem_size < file_size {
            return Err(Error::Elf("segment mem_size smaller than file_size".into()));
        }
        segments.push(LoadSegment {
            file_offset,
            vaddr,
            file_size,
            mem_size,
        });
    }

    Ok(KernelImage { entry, segments })
}

fn rd_u16(b: &[u8], off: usize) -> Result<u16> {
    b.get(off..off + 2)
        .map(|s| u16::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| Error::Elf("truncated ELF field".into()))
}

fn rd_u32(b: &[u8], off: usize) -> Result<u32> {
    b.get(off..off + 4)
        .map(|s| u32::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| Error::Elf("truncated ELF field".into()))
}

fn rd_u64(b: &[u8], off: usize) -> Result<u64> {
    b.get(off..off + 8)
        .map(|s| u64::from_le_bytes(s.try_into().unwrap()))
        .ok_or_else(|| Error::Elf("truncated ELF field".into()))
}
