mod bind_entry;
mod layout_entry;

use bind_entry::BindEntry;
use layout_entry::LayoutEntry;

use crate::{Buffer, Device};

pub struct BindGroup<'a, 'device> {
    layout_entries: Vec<LayoutEntry>,
    bind_entries: Vec<BindEntry<'a>>,
    device: &'device Device,
}

impl<'a, 'device> BindGroup<'a, 'device> {
    pub fn group(self) -> wgpu::BindGroup {
        let layout_entries = self
            .layout_entries
            .into_iter()
            .map(|le| le.into_entry())
            .collect::<Vec<_>>();
        let bind_entries = self
            .bind_entries
            .into_iter()
            .map(|be| be.into_entry())
            .collect::<Vec<_>>();
        let layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &layout_entries,
        });
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &bind_entries,
        })
    }
}

// Builder implementation

pub struct BindGroupBuilder<'a, 'device> {
    buffers: Vec<&'a Buffer>,
    layout_entries: Vec<LayoutEntry>,
    bind_entries: Vec<BindEntry<'a>>,
    counter: u32,
    device: &'device Device,
}

impl<'a, 'device> BindGroupBuilder<'a, 'device> {
    pub fn new(device: &'device Device) -> Self {
        Self {
            buffers: Vec::new(),
            layout_entries: Vec::new(),
            bind_entries: Vec::new(),
            counter: 0u32,
            device,
        }
    }

    pub fn add_input(mut self, buffer: &'a Buffer) -> Self {
        self.layout_entries.push(LayoutEntry::new(buffer));
        self.bind_entries.push(BindEntry::new(buffer, self.counter));
        self.buffers.push(buffer);
        self.counter += 1;
        self
    }

    pub fn add_inputs(mut self, buffers: &'a [Buffer]) -> Self {
        self.layout_entries.extend(buffers.into_iter().map(LayoutEntry::new));
        self.bind_entries.extend(
            buffers
                .into_iter()
                .enumerate()
                .map(|(idx, buffer)| BindEntry::new(buffer, self.counter + idx as u32)),
        );
        self.buffers.extend(buffers);
        self.counter += buffers.len() as u32;
        self
    }

    pub fn build(mut self, output: &'a Buffer) -> BindGroup<'a, 'device> {
        self.layout_entries.push(LayoutEntry::new(output));
        self.bind_entries.push(BindEntry::new(output, self.counter + 1));
        self.buffers.push(output);
        BindGroup {
            device: self.device,
            layout_entries: self.layout_entries,
            bind_entries: self.bind_entries,
        }
    }
}
