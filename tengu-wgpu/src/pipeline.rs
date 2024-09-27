use std::ops::Deref;

use crate::{Buffer, BufferUsage, Device};

const ENTRY: &str = "main";

// Compute pipeline

pub struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl Pipeline {
    pub fn new(pipeline: wgpu::ComputePipeline, bind_group: wgpu::BindGroup) -> Self {
        Self { pipeline, bind_group }
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

impl Deref for Pipeline {
    type Target = wgpu::ComputePipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

// LayoutBuilder implementation

pub struct LayoutBuilder<'a, 'device> {
    device: &'device Device,
    label: Option<String>,
    buffers: Vec<&'a Buffer>,
    layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    bind_entries: Vec<wgpu::BindGroupEntry<'a>>,
    counter: usize,
}

impl<'a, 'device> LayoutBuilder<'a, 'device> {
    pub fn new(device: &'device Device) -> Self {
        Self {
            device,
            label: None,
            buffers: Vec::new(),
            layout_entries: Vec::new(),
            bind_entries: Vec::new(),
            counter: 0,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn add_entry(mut self, buffer: &'a Buffer) -> Self {
        self.layout_entries.push(create_layout_entry(buffer));
        self.bind_entries.push(create_bind_entry(buffer, self.counter));
        self.buffers.push(buffer);
        self.counter += 1;
        self
    }

    pub fn add_entries(mut self, buffers: impl IntoIterator<Item = &'a Buffer>) -> Self {
        for buffer in buffers.into_iter() {
            self.layout_entries.push(create_layout_entry(buffer));
            self.bind_entries.push(create_bind_entry(buffer, self.counter));
            self.buffers.push(buffer);
            self.counter += 1;
        }
        self
    }

    pub fn pipeline(self) -> PipelineBuilder<'device> {
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: self.label.as_deref(),
            entries: &self.layout_entries,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &self.bind_entries,
        });
        PipelineBuilder::new(self.device, bind_group, bind_group_layout)
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

// PipelineBuilder implementation

pub struct PipelineBuilder<'device> {
    device: &'device Device,
    label: Option<String>,
    layout: wgpu::PipelineLayout,
    bind_group: wgpu::BindGroup,
}

impl<'device> PipelineBuilder<'device> {
    pub fn new(device: &'device Device, bind_group: wgpu::BindGroup, bind_group_layout: wgpu::BindGroupLayout) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        Self {
            device,
            label: None,
            layout: pipeline_layout,
            bind_group,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn build(self, shader: wgpu::ShaderModule) -> Pipeline {
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: self.label.as_deref(),
            layout: Some(&self.layout),
            module: &shader,
            entry_point: ENTRY,
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Pipeline::new(pipeline, self.bind_group)
    }
}
