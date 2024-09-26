use crate::Buffer;

pub struct BindEntry<'a> {
    entry: wgpu::BindGroupEntry<'a>,
}

impl<'a> BindEntry<'a> {
    pub fn new(buffer: &'a Buffer, idx: u32) -> Self {
        let entry = wgpu::BindGroupEntry {
            binding: idx,
            resource: buffer.as_entire_binding(),
        };
        Self { entry }
    }

    pub fn into_entry(self) -> wgpu::BindGroupEntry<'a> {
        self.entry
    }
}
