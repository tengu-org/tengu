use tengu_wgpu_util::{Device, Queue, WGPU};

use crate::tensor::TensorBuilder;

pub struct Context {
    device: Device,
    queue: Queue,
}

impl Context {
    pub async fn new() -> Self {
        let instance = WGPU::builder().backends(wgpu::Backends::PRIMARY).build();
        let (device, queue) = instance
            .adapter()
            .build()
            .await
            .expect("Failed to find an appropriate adapter")
            .device()
            .build()
            .await
            .expect("Failed to create device");
        Self { device, queue }
    }

    pub fn tensor(&self, shape: impl Into<Vec<usize>>) -> TensorBuilder {
        TensorBuilder::new(self, shape)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn compute<F>(&self, f: F)
    where
        F: FnOnce(&mut wgpu::CommandEncoder),
    {
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Command Encoder"),
        });
        f(&mut encoder);
        self.queue.submit(Some(encoder.finish()));
    }
}
