//! An image's program: loaded, and run a little. Neither the loader nor the
//! system calls the program makes may panic, whatever its bytes.
#![no_main]

#[path = "common.rs"]
mod common;

use libfuzzer_sys::fuzz_target;
use shem::container::Container;
use shem::reference::ReferenceVcpu;

fuzz_target!(|elf: &[u8]| {
    let vcpu = ReferenceVcpu::new(common::MEM);
    let argv = ["fuzz".to_string()];
    if let Ok(mut container) = Container::from_elf(vcpu, elf, &argv, &[], common::stdio()) {
        let _ = container.run(Some(common::STEPS));
    }
});
