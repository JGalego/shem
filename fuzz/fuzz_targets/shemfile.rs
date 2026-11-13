//! A Shemfile, parsed: it may be wrong, and say so, but not panic.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|text: &str| {
    let _ = shem::image::shemfile::parse(text);
});
