//! What the targets share: a container with nothing to read and nowhere to
//! write, and the echo program from the seeds, frozen at its read.
#![allow(dead_code)]

use std::sync::OnceLock;

use shem::container::{Container, Frozen, Pause, Stop};
use shem::linux::Stdio;
use shem::reference::ReferenceVcpu;

/// Instructions a container runs before the target moves on.
pub const STEPS: u64 = 10_000;
/// Guest memory, enough for the seeds' programs.
pub const MEM: u32 = 1 << 20;
/// The echo program the seeds hold: it writes, reads, writes, and exits.
pub const ECHO: &[u8] = include_bytes!("../seeds/elf/echo");

/// Standard streams with nothing to read, that take whatever is written.
pub fn stdio() -> Stdio {
    Stdio {
        stdin: Box::new(std::io::empty()),
        stdout: Box::new(std::io::sink()),
        stderr: Box::new(std::io::sink()),
    }
}

/// The echo program, frozen as it is about to read.
pub fn frozen_echo() -> &'static Frozen {
    static FROZEN: OnceLock<Frozen> = OnceLock::new();
    FROZEN.get_or_init(|| {
        let vcpu = ReferenceVcpu::new(MEM);
        let mut container = Container::from_elf(vcpu, ECHO, &["echo".into()], &[], stdio())
            .expect("the echo seed loads");
        let read = Pause::Syscall(63);
        assert_eq!(
            container.run_until(Some(STEPS), Some(read)),
            Ok(Stop::Paused)
        );
        container.freeze().expect("the echo seed freezes")
    })
}

/// Thaw `frozen` and run it a little. Neither may panic, whatever the state.
pub fn thaw_and_run(frozen: &Frozen) {
    let vcpu = ReferenceVcpu::new(frozen.machine.mem.len() as u32);
    if let Ok(mut container) = Container::thaw(vcpu, frozen, stdio()) {
        let _ = container.run(Some(container.steps.saturating_add(STEPS)));
    }
}
