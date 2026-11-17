//! The C extension: compressed (16-bit) instructions on the reference vCPU,
//! for programs built with it (the ELF header's RVC flag), and not otherwise.

mod common;

use common::*;
use shem::elf;
use shem::reference::{ReferenceVcpu, expand_compressed};
use shem::vcpu::{ExitReason, MachineState, Vcpu};

const MEM: u32 = 64 << 10;

/// Every instruction in `data/compressed.txt` as clang's assembler encodes it
/// compressed, and written out in full: what `expand_compressed` must give.
#[test]
fn every_compressed_form_expands_to_what_the_assembler_writes_in_full() {
    let table = include_str!("data/compressed.txt");
    let mut checked = 0;
    for line in table.lines().filter(|line| !line.starts_with('#')) {
        let mut fields = line.split_whitespace();
        let half = u16::from_str_radix(fields.next().unwrap(), 16).unwrap();
        let word = u32::from_str_radix(fields.next().unwrap(), 16).unwrap();
        assert_eq!(expand_compressed(half), Some(word), "{line}");
        checked += 1;
    }
    assert!(checked > 3000, "{checked} forms");
}

#[test]
fn what_has_no_rv32_form_is_illegal() {
    for (half, why) in [
        (0x0000u16, "the all-zero halfword"),
        (0x0004, "c.addi4spn with no offset"),
        (0x2000, "c.fld"),
        (0x6000, "c.flw"),
        (0xa000, "c.fsd"),
        (0xe000, "c.fsw"),
        (0x4002, "c.lwsp into x0"),
        (0x8002, "c.jr x0"),
        (0x6101, "c.addi16sp with no offset"),
        (0x6081, "c.lui with no immediate"),
        (0x9001, "c.srli with a shift past 31"),
        (0x9c01, "c.subw (RV64)"),
        (0x1002, "c.slli with a shift past 31"),
        (0x2002, "c.fldsp"),
        (0x6002, "c.flwsp"),
        (0xa002, "c.fsdsp"),
        (0xe002, "c.fswsp"),
    ] {
        assert_eq!(expand_compressed(half), None, "{why}: {half:#06x}");
    }
}

/// A vCPU with `code` (16-bit units) at 0, compressed instructions on.
fn machine(code: &[u16]) -> ReferenceVcpu {
    let mut vcpu = ReferenceVcpu::new(MEM);
    assert!(vcpu.set_compressed(true));
    let bytes: Vec<u8> = code.iter().flat_map(|half| half.to_le_bytes()).collect();
    vcpu.write_mem(0, &bytes).unwrap();
    vcpu
}

const C_EBREAK: u16 = 0x9002;

#[test]
fn compressed_and_ordinary_instructions_mix_at_any_halfword() {
    // c.li a0,5; c.addi a0,3; addi a1,a0,1 (32 bits, at offset 4); c.nop; addi a2,a1,1 (32 bits, at offset 10)
    let addi_a1 = 0x0015_0593u32;
    let addi_a2 = 0x0015_8613u32;
    let mut vcpu = machine(&[
        0x4515,
        0x050d,
        addi_a1 as u16,
        (addi_a1 >> 16) as u16,
        0x0001,
        addi_a2 as u16,
        (addi_a2 >> 16) as u16,
        C_EBREAK,
    ]);
    let exit = vcpu.run(100);
    assert_eq!(exit.reason, ExitReason::Ebreak);
    assert_eq!(vcpu.reg(10), 8);
    assert_eq!(vcpu.reg(11), 9);
    assert_eq!(vcpu.reg(12), 10);
    assert_eq!(
        exit.steps, 5,
        "each is one instruction, whatever its length"
    );
    assert_eq!(exit.pc, 14, "the exit points at the c.ebreak");
}

#[test]
fn links_and_jumps_are_by_the_length_of_the_instruction() {
    // 0: c.jal +4 (to 4, ra = 2); 2: c.ebreak (not reached); 4: c.mv a1,ra; 6: c.jalr a1 (to 2, ra = 8)
    let mut vcpu = machine(&[0x2011, C_EBREAK, 0x8586, 0x9582]);
    let exit = vcpu.run(100);
    assert_eq!(exit.reason, ExitReason::Ebreak);
    assert_eq!(vcpu.reg(11), 2, "c.jal's link is the next halfword");
    assert_eq!(vcpu.reg(1), 8, "c.jalr's is the one after it");
    assert_eq!(exit.pc, 2);
}

#[test]
fn branches_loads_and_stores_work_with_the_compact_registers_and_the_stack() {
    let mut vcpu = machine(&[
        0x451d, // c.li a0, 7
        0xc42a, // c.swsp a0, 8(sp)
        0x4622, // c.lwsp a2, 8(sp)
        0xc111, // c.beqz a0, +4: not taken, a0 is 7
        0xe111, // c.bnez a0, +4: taken
        C_EBREAK, C_EBREAK,
    ]);
    vcpu.set_reg(2, 0x1000);
    let exit = vcpu.run(100);
    assert_eq!(exit.reason, ExitReason::Ebreak);
    assert_eq!(vcpu.reg(12), 7);
    assert_eq!(
        exit.pc, 12,
        "the taken branch went over one c.ebreak, and not the one after"
    );
}

#[test]
fn an_illegal_compressed_instruction_exits_with_its_16_bits() {
    let mut vcpu = machine(&[0x0001, 0x0000]);
    let exit = vcpu.run(100);
    assert_eq!(exit.reason, ExitReason::IllegalInstruction(0));
    assert_eq!((exit.pc, exit.steps), (2, 1));
}

#[test]
fn only_odd_addresses_are_misaligned_with_compressed_instructions_and_any_but_multiples_of_four_without()
 {
    let mut vcpu = machine(&[0x0001, C_EBREAK]);
    vcpu.set_pc(1);
    assert_eq!(vcpu.run(10).reason, ExitReason::MemoryFault(1));
    // Without the extension nothing changes: a halfword boundary is a fault
    let mut plain = ReferenceVcpu::new(MEM);
    plain
        .write_mem(0, &[0x01, 0x00, 0x01, 0x00, 0x73, 0, 0x10, 0])
        .unwrap();
    plain.set_pc(2);
    assert_eq!(plain.run(10).reason, ExitReason::MemoryFault(2));
    plain.set_pc(0);
    assert_eq!(
        plain.run(10).reason,
        ExitReason::IllegalInstruction(0x0001_0001)
    );
}

#[test]
fn a_snapshot_remembers_the_extension_and_an_old_one_is_written_as_before() {
    let vcpu = machine(&[C_EBREAK]);
    let state = vcpu.snapshot();
    assert!(state.compressed);
    let encoded = state.encode();
    assert_eq!(MachineState::decode(&encoded).unwrap(), state);
    let mut other = ReferenceVcpu::new(MEM);
    other.restore(&state).unwrap();
    assert_eq!(other.run(10).reason, ExitReason::Ebreak);
    // Without it, the bytes are what version 1 always was
    let plain = ReferenceVcpu::new(MEM).snapshot();
    assert!(!plain.compressed);
    assert_eq!(&plain.encode()[4..8], &1u32.to_le_bytes());
    assert_eq!(MachineState::decode(&plain.encode()).unwrap(), plain);
    // A reset starts the next program without it
    let mut reset = machine(&[C_EBREAK]);
    reset.reset();
    assert!(!reset.snapshot().compressed);
}

#[test]
fn a_program_built_with_the_extension_runs_and_one_that_is_not_does_not_inherit_it() {
    // c.li a0,7; c.li a7,31; c.addi a7,31; c.addi a7,31 (= 93, exit); ecall
    let halves = [0x451du32, 0x48fd, 0x08fd, 0x08fd];
    let code = [
        halves[0] | halves[1] << 16,
        halves[2] | halves[3] << 16,
        ECALL,
    ];
    let program = elf_with_flags(&code, &[], 0x1);
    let (mut container, _) = start(&program, &["prog"], &[], 2 * MEM).unwrap();
    assert_eq!(container.run(Some(1000)), Ok(7));
    // The same words, built without the flag, are not compressed code: they fault
    let plain = elf_with_flags(&code, &[], 0);
    let (mut container, _) = start(&plain, &["prog"], &[], 2 * MEM).unwrap();
    assert!(container.run(Some(1000)).is_err());
}

/// A vCPU that cannot run compressed code, as the neural vCPU cannot.
struct Plain(ReferenceVcpu);

impl Vcpu for Plain {
    fn pc(&self) -> u32 {
        self.0.pc()
    }
    fn set_pc(&mut self, pc: u32) {
        self.0.set_pc(pc)
    }
    fn reg(&self, index: usize) -> u32 {
        self.0.reg(index)
    }
    fn set_reg(&mut self, index: usize, value: u32) {
        self.0.set_reg(index, value)
    }
    fn mem_size(&self) -> u32 {
        self.0.mem_size()
    }
    fn read_mem(&self, addr: u32, buf: &mut [u8]) -> Result<(), shem::vcpu::MemoryFault> {
        self.0.read_mem(addr, buf)
    }
    fn write_mem(&mut self, addr: u32, data: &[u8]) -> Result<(), shem::vcpu::MemoryFault> {
        self.0.write_mem(addr, data)
    }
    fn reset(&mut self) {
        self.0.reset()
    }
    fn run(&mut self, budget: u64) -> shem::vcpu::Exit {
        self.0.run(budget)
    }
    fn snapshot(&self) -> MachineState {
        self.0.snapshot()
    }
    fn restore(&mut self, state: &MachineState) -> Result<(), shem::vcpu::SnapshotError> {
        self.0.restore(state)
    }
}

#[test]
fn a_vcpu_that_cannot_run_compressed_code_refuses_the_program_and_says_so() {
    let program = elf_with_flags(&[ECALL], &[], 0x1);
    let mut vcpu = Plain(ReferenceVcpu::new(2 * MEM));
    assert_eq!(
        elf::load(&mut vcpu, &program),
        Err(elf::ElfError::Compressed)
    );
    let plain = elf_with_flags(&[ECALL], &[], 0);
    assert!(elf::load(&mut vcpu, &plain).is_ok());
}
