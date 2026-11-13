//! A frozen container's `container.json`, as a warm image or a committed
//! container carries it in its own files, which the containerd shim thaws:
//! here with a real program's memory and registers.
#![no_main]

#[path = "common.rs"]
mod common;

use libfuzzer_sys::fuzz_target;
use shem::container::{Frozen, FrozenHost};

fuzz_target!(|json: &[u8]| {
    let Ok(host) = serde_json::from_slice::<FrozenHost>(json) else {
        return;
    };
    let machine = common::frozen_echo().machine.clone();
    common::thaw_and_run(&Frozen { machine, host });
});
