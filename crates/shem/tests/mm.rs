use shem::elf::PAGE_SIZE;
use shem::mm::{AddressSpace, MapError};
use shem::reference::ReferenceVcpu;
use shem::vcpu::Vcpu;

const BRK: u32 = 0x1_0000;
const TOP: u32 = 0x10_0000;

fn setup() -> (AddressSpace, ReferenceVcpu) {
    (AddressSpace::new(BRK, TOP), ReferenceVcpu::new(TOP))
}

fn read_word(vcpu: &ReferenceVcpu, addr: u32) -> [u8; 4] {
    let mut word = [0xaa; 4];
    vcpu.read_mem(addr, &mut word).unwrap();
    word
}

#[test]
fn mappings_grow_down_from_the_top_and_come_back_zeroed() {
    let (mut mm, mut vcpu) = setup();
    let a = mm.map(&mut vcpu, 0, 100, false).unwrap();
    let b = mm.map(&mut vcpu, 0, PAGE_SIZE + 1, false).unwrap();
    assert_eq!(a, TOP - PAGE_SIZE);
    assert_eq!(b, TOP - 3 * PAGE_SIZE);

    vcpu.write_mem(a, &[0xff; 4]).unwrap();
    mm.unmap(a, PAGE_SIZE).unwrap();
    assert_eq!(mm.map(&mut vcpu, 0, 1, false), Ok(a));
    assert_eq!(read_word(&vcpu, a), [0; 4]);
}

#[test]
fn unmapping_the_middle_splits_a_mapping() {
    let (mut mm, mut vcpu) = setup();
    let base = mm.map(&mut vcpu, 0, 3 * PAGE_SIZE, false).unwrap();
    mm.unmap(base + PAGE_SIZE, PAGE_SIZE).unwrap();
    assert_eq!(mm.map(&mut vcpu, 0, PAGE_SIZE, false), Ok(base + PAGE_SIZE));
    assert_eq!(
        mm.map(&mut vcpu, 0, 2 * PAGE_SIZE, false),
        Ok(base - 2 * PAGE_SIZE)
    );
}

#[test]
fn heap_and_mappings_cannot_cross() {
    let (mut mm, mut vcpu) = setup();
    let low = mm.map(&mut vcpu, 0, TOP - BRK - PAGE_SIZE, false).unwrap();
    assert_eq!(low, BRK + PAGE_SIZE);
    assert_eq!(mm.set_brk(&mut vcpu, low + 1), BRK);
    assert_eq!(mm.set_brk(&mut vcpu, low), low);
    assert_eq!(
        mm.map(&mut vcpu, 0, PAGE_SIZE, false),
        Err(MapError::NoMemory)
    );
}

#[test]
fn fixed_mappings_replace_what_was_there() {
    let (mut mm, mut vcpu) = setup();
    let base = mm.map(&mut vcpu, 0, 2 * PAGE_SIZE, false).unwrap();
    vcpu.write_mem(base, &[7; 4]).unwrap();
    assert_eq!(mm.map(&mut vcpu, base, PAGE_SIZE, true), Ok(base));
    assert_eq!(read_word(&vcpu, base), [0; 4]);
    assert_eq!(
        mm.map(&mut vcpu, base + 1, PAGE_SIZE, true),
        Err(MapError::Invalid)
    );
    let in_the_image = BRK - PAGE_SIZE;
    assert_eq!(
        mm.map(&mut vcpu, in_the_image, PAGE_SIZE, true),
        Err(MapError::NoMemory)
    );
}

#[test]
fn rejects_malformed_requests() {
    let (mut mm, mut vcpu) = setup();
    assert_eq!(mm.map(&mut vcpu, 0, 0, false), Err(MapError::Invalid));
    assert_eq!(
        mm.map(&mut vcpu, 0, u32::MAX, false),
        Err(MapError::NoMemory)
    );
    assert_eq!(mm.unmap(1, PAGE_SIZE), Err(MapError::Invalid));
    assert_eq!(mm.unmap(0, 0), Err(MapError::Invalid));
}
