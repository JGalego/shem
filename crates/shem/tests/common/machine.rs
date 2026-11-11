//! Machine-mode programs for VM mode, shared by the VM and neural tests.

use super::*;

pub const RAM: u32 = 64 << 10;
pub const MTVEC: u32 = 0x305;
pub const MSTATUS: u32 = 0x300;
pub const MIE: u32 = 0x304;
pub const MEPC: u32 = 0x341;
pub const MCAUSE: u32 = 0x342;
pub const MTVAL: u32 = 0x343;
pub const MSCRATCH: u32 = 0x340;
pub const MISA: u32 = 0x301;
pub const UART: u32 = 0x1000_0000;
pub const MTIMECMP: u32 = 0x1100_4000;
pub const MTIME: u32 = 0x1100_bff8;
pub const SYSCON: u32 = 0x1110_0000;

/// Print the low byte of `reg` on the UART.
pub fn putc(reg: u32) -> [u32; 3] {
    let [upper, low] = li(T1, UART);
    [upper, low, sb(reg, T1, 0)]
}

/// Print `text` on the UART.
pub fn print(text: &str) -> Vec<u32> {
    text.bytes()
        .flat_map(|b| [li(A0, b as u32).to_vec(), putc(A0).to_vec()].concat())
        .collect()
}

/// Power the machine off through the syscon.
pub fn poweroff() -> Vec<u32> {
    program![li(T1, SYSCON), li(T2, 0x5555), sw(T2, T1, 0)]
}

/// `main`, given the handler's offset from the start, followed by `handler`;
/// main installs the handler as the trap vector first.
pub fn with_handler(main: impl Fn(i32) -> Vec<u32>, handler: Vec<u32>) -> Vec<u8> {
    let install = |offset: i32| {
        let mut code = program![auipc(T0, 0), addi(T0, T0, offset), csr(1, ZERO, MTVEC, T0)];
        code.extend(main(offset));
        code
    };
    let offset = install(0).len() as i32 * 4;
    bytes(&[install(offset), handler].concat())
}

/// Arm the timer `ticks` ahead, enable its interrupt, then run `wait` until
/// the handler sets a2. (a1 starts out holding the device tree's address.)
pub fn timer_program(ticks: i32, wait: Vec<u32>) -> Vec<u8> {
    let main = move |_| {
        program![
            li(T1, MTIME),
            lw(T2, T1, 0),
            addi(T2, T2, ticks),
            li(T1, MTIMECMP),
            sw(T2, T1, 0),
            sw(ZERO, T1, 4),
            addi(T2, ZERO, 0x80),
            csr(1, ZERO, MIE, T2),    // mie.MTIE
            csr(6, ZERO, MSTATUS, 8), // mstatus.MIE
        ]
        .into_iter()
        .chain(wait.clone())
        .chain(program![addi(A2, A2, 0x40), putc(A2), poweroff()])
        .collect()
    };
    let handler = program![
        csr(2, A2, MCAUSE, ZERO), // 0x80000007
        andi_(A2, A2, 0xf),       // 7
        csr(1, ZERO, MIE, ZERO),  // no more timer interrupts
        MRET,
    ];
    with_handler(main, handler)
}

pub fn andi_(rd: u32, rs1: u32, imm: i32) -> u32 {
    i(0x13, 7, rd, rs1, imm)
}
