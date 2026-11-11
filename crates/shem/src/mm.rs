//! The guest's address space, managed by the host.
//!
//! Layout, from low to high addresses: the program image, the heap growing up
//! from the program break, anonymous mappings growing down from `top`, and the
//! stack above `top`. The heap and the mappings share the gap between them.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::elf::PAGE_SIZE;
use crate::vcpu::Vcpu;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MapError {
    /// The request is malformed: zero length, or a misaligned address.
    Invalid,
    /// There is no room for the mapping.
    NoMemory,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AddressSpace {
    brk_start: u32,
    brk: u32,
    top: u32,
    /// Anonymous mappings as `start -> end`, page-aligned and non-overlapping.
    maps: BTreeMap<u32, u32>,
}

impl AddressSpace {
    pub fn new(brk: u32, top: u32) -> Self {
        Self {
            brk_start: brk,
            brk,
            top,
            maps: BTreeMap::new(),
        }
    }

    pub fn brk(&self) -> u32 {
        self.brk
    }

    /// Bytes in use below the stack: the program image, the heap and the
    /// mappings.
    pub fn in_use(&self) -> u64 {
        let mapped: u64 = self
            .maps
            .iter()
            .map(|(start, end)| (end - start) as u64)
            .sum();
        self.brk as u64 + mapped
    }

    /// Move the program break if the request is in range, and return where the
    /// break now is, as Linux does.
    pub fn set_brk(&mut self, vcpu: &mut dyn Vcpu, addr: u32) -> u32 {
        let limit = self.maps.keys().next().copied().unwrap_or(self.top);
        if (self.brk_start..=limit).contains(&addr) {
            // Memory past the break may hold data from before it last shrank.
            let grown = addr.saturating_sub(self.brk) as usize;
            if grown == 0 || vcpu.write_mem(self.brk, &vec![0; grown]).is_ok() {
                self.brk = addr;
            }
        }
        self.brk
    }

    /// Map `len` bytes of zeroed memory and return its address. With `fixed`,
    /// map exactly at `addr`, replacing whatever was there.
    pub fn map(
        &mut self,
        vcpu: &mut dyn Vcpu,
        addr: u32,
        len: u32,
        fixed: bool,
    ) -> Result<u32, MapError> {
        if len == 0 {
            return Err(MapError::Invalid);
        }
        let len = page_up(len).ok_or(MapError::NoMemory)?;
        let heap_end = page_up(self.brk).ok_or(MapError::NoMemory)?;
        let start = if fixed {
            if !addr.is_multiple_of(PAGE_SIZE) {
                return Err(MapError::Invalid);
            }
            match addr.checked_add(len) {
                Some(end) if addr >= heap_end && end <= self.top => {}
                _ => return Err(MapError::NoMemory),
            }
            self.unmap(addr, len)?;
            addr
        } else {
            self.find_gap(heap_end, len).ok_or(MapError::NoMemory)?
        };
        vcpu.write_mem(start, &vec![0; len as usize])
            .map_err(|_| MapError::NoMemory)?;
        self.maps.insert(start, start + len);
        Ok(start)
    }

    /// Remove any mappings in `[addr, addr + len)`, splitting ones that straddle it.
    pub fn unmap(&mut self, addr: u32, len: u32) -> Result<(), MapError> {
        let len = page_up(len).ok_or(MapError::Invalid)?;
        let end = addr.checked_add(len).ok_or(MapError::Invalid)?;
        if !addr.is_multiple_of(PAGE_SIZE) || len == 0 {
            return Err(MapError::Invalid);
        }
        let overlapping: Vec<(u32, u32)> = self
            .maps
            .range(..end)
            .filter(|&(_, &e)| e > addr)
            .map(|(&s, &e)| (s, e))
            .collect();
        for (s, e) in overlapping {
            self.maps.remove(&s);
            if s < addr {
                self.maps.insert(s, addr);
            }
            if e > end {
                self.maps.insert(end, e);
            }
        }
        Ok(())
    }

    /// The highest free range of `len` bytes between the heap and `top`.
    fn find_gap(&self, floor: u32, len: u32) -> Option<u32> {
        let mut ceiling = self.top;
        for (&start, &end) in self.maps.iter().rev() {
            if ceiling - end >= len {
                break;
            }
            ceiling = start;
        }
        ceiling.checked_sub(len).filter(|&start| start >= floor)
    }
}

fn page_up(value: u32) -> Option<u32> {
    value.checked_next_multiple_of(PAGE_SIZE)
}
