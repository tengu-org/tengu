use tengu_wgpu_util::{Device, Queue, WGPU};

use crate::tensor::TensorBuilder;

pub struct Context {
    device: Device,
    queue: Queue,
}

impl Context {
    pub async fn new() -> Self {
        let instance = WGPU::instance().with_backends(wgpu::Backends::PRIMARY);
        let adapter = instance
            .builder()
            .with_default_adapter()
            .await
            .expect("Failed to find an appropriate adapter");
        let (device, queue) = adapter.request_device().await.expect("Failed to create device");
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
}
