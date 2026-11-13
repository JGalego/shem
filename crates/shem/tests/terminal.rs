//! Signals typed at a terminal, which go to the foreground processes of
//! whichever container the host process runs: in a process apart from the
//! other tests.

mod common;

use std::sync::atomic::Ordering;

use common::*;
use shem::container::TERMINAL_SIGNAL;

const MEM: u32 = 1 << 20;

#[test]
fn typed_signals_reach_init_and_return_from_no_system_call() {
    // Ctrl-C ends an init that does not handle SIGINT
    let (mut spin, _) = start(&elf(&program![jal(ZERO, 0)], &[]), &["spin"], b"", MEM).unwrap();
    TERMINAL_SIGNAL.store(2, Ordering::SeqCst);
    assert_eq!(spin.run(Some(1_000_000)).unwrap(), 128 + 2);

    // A signal it ignores leaves it as it was: it made no call to return from
    let code = program![addi(A0, ZERO, 7), addi(A7, ZERO, 93), ECALL];
    let (mut seven, _) = start(&elf(&code, &[]), &["seven"], b"", MEM).unwrap();
    TERMINAL_SIGNAL.store(28, Ordering::SeqCst); // SIGWINCH
    assert_eq!(seven.run(Some(1_000_000)).unwrap(), 7);
}
