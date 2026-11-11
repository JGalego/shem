//! A flattened device tree writer: the blob a kernel reads at boot to learn
//! what machine it is on (devicetree specification, version 17).

const MAGIC: u32 = 0xd00d_feed;
const BEGIN_NODE: u32 = 1;
const END_NODE: u32 = 2;
const PROP: u32 = 3;
const END: u32 = 9;

#[derive(Default)]
pub struct Fdt {
    structure: Vec<u8>,
    strings: Vec<u8>,
}

impl Fdt {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn begin(&mut self, name: &str) -> &mut Self {
        self.token(BEGIN_NODE);
        self.structure.extend_from_slice(name.as_bytes());
        self.structure.push(0);
        self.pad();
        self
    }

    pub fn end(&mut self) -> &mut Self {
        self.token(END_NODE);
        self
    }

    pub fn prop(&mut self, name: &str, value: &[u8]) -> &mut Self {
        let offset = self.string(name);
        self.token(PROP);
        self.token(value.len() as u32);
        self.token(offset);
        self.structure.extend_from_slice(value);
        self.pad();
        self
    }

    pub fn empty(&mut self, name: &str) -> &mut Self {
        self.prop(name, &[])
    }

    pub fn cells(&mut self, name: &str, cells: &[u32]) -> &mut Self {
        let value: Vec<u8> = cells.iter().flat_map(|c| c.to_be_bytes()).collect();
        self.prop(name, &value)
    }

    /// A string, or several, each NUL-terminated.
    pub fn strings(&mut self, name: &str, values: &[&str]) -> &mut Self {
        let mut value = Vec::new();
        for s in values {
            value.extend_from_slice(s.as_bytes());
            value.push(0);
        }
        self.prop(name, &value)
    }

    pub fn finish(mut self) -> Vec<u8> {
        self.token(END);
        let header_size = 40u32;
        let reserve_map = header_size;
        let structure = reserve_map + 16; // one empty reservation entry
        let strings = structure + self.structure.len() as u32;
        let total = strings + self.strings.len() as u32;
        let mut blob = Vec::with_capacity(total as usize);
        for word in [
            MAGIC,
            total,
            structure,
            strings,
            reserve_map,
            17, // version
            16, // last compatible version
            0,  // boot cpu
            self.strings.len() as u32,
            self.structure.len() as u32,
        ] {
            blob.extend_from_slice(&word.to_be_bytes());
        }
        blob.extend_from_slice(&[0; 16]);
        blob.extend_from_slice(&self.structure);
        blob.extend_from_slice(&self.strings);
        blob
    }

    fn token(&mut self, value: u32) {
        self.structure.extend_from_slice(&value.to_be_bytes());
    }

    fn pad(&mut self) {
        while !self.structure.len().is_multiple_of(4) {
            self.structure.push(0);
        }
    }

    /// The offset of `name` in the strings block, adding it if new.
    fn string(&mut self, name: &str) -> u32 {
        let needle: Vec<u8> = name.bytes().chain([0]).collect();
        let found = self
            .strings
            .windows(needle.len())
            .enumerate()
            .find(|(i, window)| {
                *window == needle.as_slice() && (*i == 0 || self.strings[i - 1] == 0)
            });
        if let Some((offset, _)) = found {
            return offset as u32;
        }
        let offset = self.strings.len() as u32;
        self.strings.extend_from_slice(&needle);
        offset
    }
}
