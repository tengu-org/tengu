use crate::{Buffer, Device};

pub struct Encoder {
    encoder: wgpu::CommandEncoder,
}

impl Encoder {
    pub fn new(device: &Device, label: &str) -> Self {
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        Self { encoder }
    }

    pub fn pass(mut self, label: &str, call: impl FnOnce(wgpu::ComputePass)) -> Self {
        let compute_pass = self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        call(compute_pass);
        self
    }

    pub fn readout(mut self, call: impl FnOnce(&mut Encoder)) -> Self {
        call(&mut self);
        self
    }

    pub fn copy_buffer(&mut self, source: &Buffer, destination: &Buffer) {
        let size = source.size();
        self.encoder.copy_buffer_to_buffer(source, 0, destination, 0, size);
    }

    pub fn finish(self) -> wgpu::CommandBuffer {
        self.encoder.finish()
    }
}
