//! A frozen container's `machine.state`: decoded, encoded back the same, and
//! thawed with a real program's host state, however little memory it has.
#![no_main]

#[path = "common.rs"]
mod common;

use libfuzzer_sys::fuzz_target;
use shem::container::Frozen;
use shem::vcpu::MachineState;

fuzz_target!(|data: &[u8]| {
    let Ok(machine) = MachineState::decode(data) else {
        return;
    };
    assert_eq!(
        MachineState::decode(&machine.encode()).as_ref(),
        Ok(&machine)
    );
    let host = common::frozen_echo().host.clone();
    common::thaw_and_run(&Frozen { machine, host });
});
