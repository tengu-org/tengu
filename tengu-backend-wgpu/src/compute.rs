use tengu_backend::Backend;
use tengu_wgpu::{Device, Pipeline};

use crate::processor::Processor;
use crate::Backend as WGPUBackend;

const WORKGROUP_SIZE: u32 = 64;

pub struct Compute<'a> {
    device: &'a Device,
    label: String,
    pass: wgpu::ComputePass<'a>,
}

impl<'a> Compute<'a> {
    pub fn new(device: &'a Device, label: impl Into<String>, pass: wgpu::ComputePass<'a>) -> Self {
        Self {
            device,
            label: label.into(),
            pass,
        }
    }

    fn pipeline(&self, processor: &Processor<'_>) -> Pipeline {
        let shader = self.device.shader(&self.label, processor.shader());
        let buffers = processor.sources().map(|source| source.buffer());
        self.device
            .layout()
            .add_entries(buffers)
            .pipeline(&self.label)
            .build(shader)
    }
}

impl<'a> tengu_backend::Compute for Compute<'a> {
    type Backend = WGPUBackend;

    fn commit(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>) {
        let pipeline = self.pipeline(processor);
        let workgroup_count = processor.element_count() as u32 / WORKGROUP_SIZE + 1;
        self.pass.set_pipeline(&pipeline);
        self.pass.set_bind_group(0, pipeline.bind_group(), &[]);
        self.pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
}
