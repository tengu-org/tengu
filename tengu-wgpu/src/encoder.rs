use crate::{Buffer, Device};

pub struct Encoder {
    encoder: wgpu::CommandEncoder,
}

impl Encoder {
    pub fn new(device: &Device, label: &str) -> Self {
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        Self { encoder }
    }

    pub fn pass<F>(mut self, label: &str, commands: F) -> Self
    where
        F: FnOnce(&mut wgpu::ComputePass),
    {
        let mut compute_pass = self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        commands(&mut compute_pass);
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
