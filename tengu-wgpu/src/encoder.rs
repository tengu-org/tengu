use crate::{Buffer, Device};

pub struct Encoder<'device> {
    device: &'device Device,
    encoder: wgpu::CommandEncoder,
}

impl<'device> Encoder<'device> {
    pub fn new(device: &'device Device) -> Self {
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        Self { device, encoder }
    }

    pub(crate) fn compute<F>(mut self, label: &str, commands: F) -> Self
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

    pub fn finish(self) {
        let commands = self.encoder.finish();
        self.device.submit(commands);
    }
}
