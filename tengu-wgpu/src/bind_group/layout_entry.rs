use crate::{Buffer, BufferUsage};

pub struct LayoutEntry {
    entry: wgpu::BindGroupLayoutEntry,
}

impl LayoutEntry {
    pub fn new(buffer: &Buffer) -> Self {
        let read_only = match buffer.usage() {
            BufferUsage::Read => true,
            BufferUsage::Write => false,
            BufferUsage::ReadWrite => false,
            BufferUsage::Staging => panic!("staging buffers should not belong to a bind group"),
        };
        let entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        Self { entry }
    }

    pub fn into_entry(self) -> wgpu::BindGroupLayoutEntry {
        self.entry
    }
}
