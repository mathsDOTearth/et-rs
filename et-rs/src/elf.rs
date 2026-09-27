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

    // The whole program-header table must lie within the image. Checked
    // arithmetic rejects a hostile `e_phoff` near `u64::MAX` rather than
    // wrapping (release) or panicking (debug).
    let table_end = phnum
        .checked_mul(PHENT_SIZE)
        .and_then(|len| phoff.checked_add(len))
        .ok_or_else(|| Error::Elf("program header table range overflows".into()))?;
    if table_end > image.len() {
        return Err(Error::Elf("program header table out of bounds".into()));
    }

    let mut segments: Vec<LoadSegment> = Vec::new();
    for i in 0..phnum {
        let base = phoff + i * PHENT_SIZE;
        // ELF64 program-header field offsets (System V ABI, Table 5-2).
        let p_type = rd_u32(image, base)?; // offset  0: p_type
        if p_type != PT_LOAD {
            continue;
        }
        let file_offset = rd_u64(image, base + 8)?; // offset  8: p_offset
        let vaddr = rd_u64(image, base + 16)?; // offset 16: p_vaddr
        // offset 24: p_paddr (unused)
        let file_size = rd_u64(image, base + 32)?; // offset 32: p_filesz
        let mem_size = rd_u64(image, base + 40)?; // offset 40: p_memsz

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
        let mem_end = vaddr
            .checked_add(mem_size)
            .ok_or_else(|| Error::Elf("segment memory range overflows".into()))?;
        // Overlapping segments would make the final device contents depend on
        // load order; no well-formed kernel image contains them.
        if let Some(other) = segments.iter().find(|s| {
            mem_size > 0 && s.mem_size > 0 && vaddr < s.vaddr + s.mem_size && s.vaddr < mem_end
        }) {
            return Err(Error::Elf(format!(
                "segment [{vaddr:#x}, {mem_end:#x}) overlaps segment at {:#x}",
                other.vaddr
            )));
        }
        segments.push(LoadSegment {
            file_offset,
            vaddr,
            file_size,
            mem_size,
        });
    }

    // The entry point must lie inside a loaded segment; otherwise the launch
    // would begin executing whatever happens to occupy that DRAM. An image
    // with no PT_LOAD segments loads nothing and is accepted as-is (the host
    // test doubles rely on this to exercise launch without DMA).
    if !segments.is_empty()
        && !segments
            .iter()
            .any(|s| entry >= s.vaddr && entry < s.vaddr + s.mem_size)
    {
        return Err(Error::Elf(format!(
            "entry point {entry:#x} lies outside every PT_LOAD segment"
        )));
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

#[cfg(test)]
mod tests {
    use super::*;

    const PHOFF: usize = 64;

    /// An RV64 ELF image whose `PT_LOAD` segments are given as
    /// `(vaddr, file_size, mem_size)`; file contents are zero-filled and follow
    /// the program-header table.
    fn image(entry: u64, segments: &[(u64, u64, u64)]) -> Vec<u8> {
        let data_off = PHOFF + segments.len() * PHENT_SIZE;
        let file_total: u64 = segments.iter().map(|s| s.1).sum();
        let mut elf = vec![0u8; data_off + file_total as usize];
        elf[0..4].copy_from_slice(b"\x7fELF");
        elf[4] = 2;
        elf[5] = 1;
        elf[16..18].copy_from_slice(&2u16.to_le_bytes());
        elf[18..20].copy_from_slice(&EM_RISCV.to_le_bytes());
        elf[24..32].copy_from_slice(&entry.to_le_bytes());
        elf[32..40].copy_from_slice(&(PHOFF as u64).to_le_bytes());
        elf[54..56].copy_from_slice(&(PHENT_SIZE as u16).to_le_bytes());
        elf[56..58].copy_from_slice(&(segments.len() as u16).to_le_bytes());
        let mut file_offset = data_off as u64;
        for (i, &(vaddr, file_size, mem_size)) in segments.iter().enumerate() {
            let ph = PHOFF + i * PHENT_SIZE;
            elf[ph..ph + 4].copy_from_slice(&PT_LOAD.to_le_bytes());
            elf[ph + 8..ph + 16].copy_from_slice(&file_offset.to_le_bytes());
            elf[ph + 16..ph + 24].copy_from_slice(&vaddr.to_le_bytes());
            elf[ph + 32..ph + 40].copy_from_slice(&file_size.to_le_bytes());
            elf[ph + 40..ph + 48].copy_from_slice(&mem_size.to_le_bytes());
            file_offset += file_size;
        }
        elf
    }

    fn elf_error(image: &[u8]) -> String {
        match parse(image) {
            Err(Error::Elf(msg)) => msg,
            other => panic!("expected Error::Elf, got {other:?}"),
        }
    }

    #[test]
    fn accepts_well_formed_image() {
        let parsed = parse(&image(0x1000, &[(0x1000, 16, 32), (0x2000, 8, 8)])).unwrap();
        assert_eq!(parsed.entry, 0x1000);
        assert_eq!(parsed.segments.len(), 2);
        assert_eq!(parsed.segments[0].mem_size, 32);
    }

    #[test]
    fn rejects_program_header_offset_overflow() {
        let mut elf = image(0x1000, &[(0x1000, 8, 8)]);
        elf[32..40].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(elf_error(&elf).contains("overflows"));
    }

    #[test]
    fn rejects_program_header_table_past_end() {
        let mut elf = image(0x1000, &[(0x1000, 8, 8)]);
        elf[56..58].copy_from_slice(&100u16.to_le_bytes());
        assert!(elf_error(&elf).contains("out of bounds"));
    }

    #[test]
    fn rejects_segment_memory_overflow() {
        let elf = image(u64::MAX - 4, &[(u64::MAX - 4, 0, 16)]);
        assert!(elf_error(&elf).contains("memory range overflows"));
    }

    #[test]
    fn rejects_overlapping_segments() {
        let elf = image(0x1000, &[(0x1000, 8, 0x100), (0x1080, 8, 8)]);
        assert!(elf_error(&elf).contains("overlaps"));
    }

    #[test]
    fn rejects_entry_outside_segments() {
        let elf = image(0x9000, &[(0x1000, 8, 8)]);
        assert!(elf_error(&elf).contains("entry point"));
    }
}
