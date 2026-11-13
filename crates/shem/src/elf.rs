//! Load static RV32 Linux executables and build the initial process stack.

use thiserror::Error;

use crate::vcpu::{MemoryFault, Vcpu};

const EM_RISCV: u16 = 243;
const ET_EXEC: u16 = 2;
const PT_LOAD: u32 = 1;
const PT_INTERP: u32 = 3;
const EF_RISCV_RVC: u32 = 0x1;
const EF_RISCV_FLOAT_ABI: u32 = 0x6;
const PHDR_SIZE: u32 = 32;
pub const PAGE_SIZE: u32 = 4096;

const AT_NULL: u32 = 0;
const AT_PHDR: u32 = 3;
const AT_PHENT: u32 = 4;
const AT_PHNUM: u32 = 5;
const AT_PAGESZ: u32 = 6;
const AT_ENTRY: u32 = 9;
const AT_HWCAP: u32 = 16;
const AT_RANDOM: u32 = 25;
/// One bit per single-letter ISA extension: I and M.
const HWCAP_RV32IM: u32 = (1 << (b'I' - b'A')) | (1 << (b'M' - b'A'));

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ElfError {
    #[error("not an ELF file")]
    NotElf,
    #[error("not a 32-bit little-endian RISC-V executable")]
    WrongArch,
    #[error("only static, non-PIE executables are supported (link with -static -no-pie)")]
    NotStatic,
    #[error("compressed instructions are not supported (build with -march=rv32im)")]
    Compressed,
    #[error("hardware floating-point ABIs are not supported (build with -mabi=ilp32)")]
    FloatAbi,
    #[error("malformed ELF file")]
    Malformed,
    #[error("segment at {0:#010x} does not fit in guest memory")]
    DoesNotFit(u32),
}

/// What the loader learned about an executable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub entry: u32,
    /// First page above the highest loaded segment: where the heap starts.
    pub brk: u32,
    /// Guest address of the program headers, if a loaded segment covers them.
    pub phdr: Option<u32>,
    pub phnum: u32,
    segments: Vec<Segment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Segment {
    vaddr: u32,
    offset: usize,
    filesz: u32,
    memsz: u32,
}

/// Check that `data` is an executable this loader can place in `mem_size`
/// bytes of guest memory, and work out where everything goes.
pub fn parse(data: &[u8], mem_size: u32) -> Result<Image, ElfError> {
    if data.get(..4) != Some(b"\x7fELF".as_slice()) {
        return Err(ElfError::NotElf);
    }
    if data.get(4..6) != Some([1, 1].as_slice()) || u16_at(data, 18)? != EM_RISCV {
        return Err(ElfError::WrongArch);
    }
    if u16_at(data, 16)? != ET_EXEC {
        return Err(ElfError::NotStatic);
    }
    let flags = u32_at(data, 36)?;
    if flags & EF_RISCV_RVC != 0 {
        return Err(ElfError::Compressed);
    }
    if flags & EF_RISCV_FLOAT_ABI != 0 {
        return Err(ElfError::FloatAbi);
    }

    let entry = u32_at(data, 24)?;
    let phoff = u32_at(data, 28)?;
    let phentsize = u16_at(data, 42)? as u32;
    let phnum = u16_at(data, 44)? as u32;
    // As Linux does, take only 32-bit program headers, a page of them at
    // most: each can fill guest memory, so thousands would take ages to load
    if phentsize != PHDR_SIZE || phnum * phentsize > PAGE_SIZE {
        return Err(ElfError::Malformed);
    }
    let mut top = 0u32;
    let mut phdr = None;
    let mut segments = Vec::new();
    for i in 0..phnum {
        let ph = phoff as usize + (i * phentsize) as usize;
        let [p_type, offset, vaddr, _paddr, filesz, memsz] =
            std::array::from_fn(|field| u32_at(data, ph + 4 * field));
        match p_type? {
            PT_INTERP => return Err(ElfError::NotStatic),
            PT_LOAD => {}
            _ => continue,
        }
        let (offset, vaddr, filesz, memsz) = (offset?, vaddr?, filesz?, memsz?);
        if memsz < filesz || data.len() < offset as usize + filesz as usize {
            return Err(ElfError::Malformed);
        }
        let end = match vaddr.checked_add(memsz) {
            Some(end) if end <= mem_size => end,
            _ => return Err(ElfError::DoesNotFit(vaddr)),
        };
        if (offset..offset + filesz).contains(&phoff) {
            phdr = Some(vaddr + (phoff - offset));
        }
        top = top.max(end);
        segments.push(Segment {
            vaddr,
            offset: offset as usize,
            filesz,
            memsz,
        });
    }
    let brk = top
        .checked_next_multiple_of(PAGE_SIZE)
        .ok_or(ElfError::Malformed)?;
    Ok(Image {
        entry,
        brk,
        phdr,
        phnum,
        segments,
    })
}

/// Place the executable `data` in guest memory.
pub fn load(vcpu: &mut dyn Vcpu, data: &[u8]) -> Result<Image, ElfError> {
    let image = parse(data, vcpu.mem_size())?;
    for segment in &image.segments {
        let fits = |_: MemoryFault| ElfError::DoesNotFit(segment.vaddr);
        let bytes = &data[segment.offset..segment.offset + segment.filesz as usize];
        vcpu.write_mem(segment.vaddr, bytes).map_err(fits)?;
        let bss = vec![0; (segment.memsz - segment.filesz) as usize];
        vcpu.write_mem(segment.vaddr + segment.filesz, &bss)
            .map_err(fits)?;
    }
    Ok(image)
}

/// Lay out argc, argv, envp and the auxiliary vector below `stack_top`, the way
/// the Linux ABI expects at process entry, and return the initial stack pointer.
pub fn setup_stack(
    vcpu: &mut dyn Vcpu,
    image: &Image,
    stack_top: u32,
    argv: &[impl AsRef<[u8]>],
    envp: &[impl AsRef<[u8]>],
) -> Result<u32, MemoryFault> {
    let mut sp = stack_top;
    // AT_RANDOM normally seeds stack canaries. Fixed bytes keep runs deterministic.
    let random = push(vcpu, &mut sp, &[0; 16])?;
    let arg_ptrs = push_strings(vcpu, &mut sp, argv)?;
    let env_ptrs = push_strings(vcpu, &mut sp, envp)?;

    let mut words = vec![argv.len() as u32];
    words.extend(&arg_ptrs);
    words.push(0);
    words.extend(&env_ptrs);
    words.push(0);
    if let Some(phdr) = image.phdr {
        words.extend([AT_PHDR, phdr, AT_PHENT, PHDR_SIZE, AT_PHNUM, image.phnum]);
    }
    words.extend([
        AT_PAGESZ,
        PAGE_SIZE,
        AT_ENTRY,
        image.entry,
        AT_HWCAP,
        HWCAP_RV32IM,
        AT_RANDOM,
        random,
        AT_NULL,
        0,
    ]);

    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let fault = MemoryFault {
        addr: sp,
        len: bytes.len(),
    };
    sp = sp.checked_sub(bytes.len() as u32).ok_or(fault)? & !15;
    vcpu.write_mem(sp, &bytes)?;
    Ok(sp)
}

fn push_strings(
    vcpu: &mut dyn Vcpu,
    sp: &mut u32,
    list: &[impl AsRef<[u8]>],
) -> Result<Vec<u32>, MemoryFault> {
    list.iter()
        .map(|s| push(vcpu, sp, &[s.as_ref(), &[0]].concat()))
        .collect()
}

fn push(vcpu: &mut dyn Vcpu, sp: &mut u32, bytes: &[u8]) -> Result<u32, MemoryFault> {
    let fault = MemoryFault {
        addr: *sp,
        len: bytes.len(),
    };
    *sp = sp.checked_sub(bytes.len() as u32).ok_or(fault)?;
    vcpu.write_mem(*sp, bytes)?;
    Ok(*sp)
}

fn u16_at(data: &[u8], offset: usize) -> Result<u16, ElfError> {
    let bytes = data.get(offset..offset + 2).ok_or(ElfError::Malformed)?;
    Ok(u16::from_le_bytes(bytes.try_into().unwrap()))
}

fn u32_at(data: &[u8], offset: usize) -> Result<u32, ElfError> {
    let bytes = data.get(offset..offset + 4).ok_or(ElfError::Malformed)?;
    Ok(u32::from_le_bytes(bytes.try_into().unwrap()))
}
