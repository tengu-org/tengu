use std::ops::Deref;

use crate::Device;

pub struct Queue {
    queue: wgpu::Queue,
}

impl Queue {
    pub fn new(queue: wgpu::Queue) -> Self {
        Self { queue }
    }

    pub fn compute<F>(&self, device: &Device, f: F)
    where
        F: FnOnce(&mut wgpu::CommandEncoder),
    {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });
        f(&mut encoder);
        self.queue.submit(Some(encoder.finish()));
    }
}

impl Deref for Queue {
    type Target = wgpu::Queue;
    fn deref(&self) -> &Self::Target {
        &self.queue
    }
}
