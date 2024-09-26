use crate::{Buffer, BufferUsage, Device};

pub struct BindGroup<'a, 'device> {
    buffers: Vec<&'a Buffer>,
    layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    bind_entries: Vec<wgpu::BindGroupEntry<'a>>,
    counter: usize,
    device: &'device Device,
}

impl<'a, 'device> BindGroup<'a, 'device> {
    pub fn new(device: &'device Device) -> Self {
        Self {
            buffers: Vec::new(),
            layout_entries: Vec::new(),
            bind_entries: Vec::new(),
            counter: 0,
            device,
        }
    }

    pub fn add_entry(mut self, buffer: &'a Buffer) -> Self {
        self.layout_entries.push(create_layout_entry(buffer));
        self.bind_entries.push(create_bind_entry(buffer, self.counter));
        self.buffers.push(buffer);
        self.counter += 1;
        self
    }

    pub fn add_entries(mut self, buffers: &'a [Buffer]) -> Self {
        self.layout_entries.extend(buffers.into_iter().map(create_layout_entry));
        self.bind_entries.extend(
            buffers
                .into_iter()
                .enumerate()
                .map(|(idx, buffer)| create_bind_entry(buffer, self.counter + idx)),
        );
        self.buffers.extend(buffers);
        self.counter += buffers.len();
        self
    }

    pub fn build(self) -> wgpu::BindGroup {
        let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &self.layout_entries,
        });
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &self.bind_entries,
        })
    }
}

fn create_layout_entry(buffer: &Buffer) -> wgpu::BindGroupLayoutEntry {
    let read_only = match buffer.usage() {
        BufferUsage::Read => true,
        BufferUsage::Write => false,
        BufferUsage::ReadWrite => false,
        BufferUsage::Staging => panic!("staging buffers should not belong to a bind group"),
    };
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn create_bind_entry(buffer: &Buffer, idx: usize) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry {
        binding: idx as u32,
        resource: buffer.as_entire_binding(),
    }
}
