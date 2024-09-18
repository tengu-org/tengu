use std::ops::Deref;

use crate::{Error, Queue};

pub struct Device {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
}

impl Device {
    pub fn new(adapter: wgpu::Adapter, device: wgpu::Device) -> Device {
        Self { adapter, device }
    }

    pub fn adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }
}

impl Deref for Device {
    type Target = wgpu::Device;
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

// Device builder implementation.

pub struct DeviceBuilder {
    adapter: wgpu::Adapter,
    features: wgpu::Features,
    limits: wgpu::Limits,
}

impl DeviceBuilder {
    pub fn new(adapter: wgpu::Adapter) -> Self {
        DeviceBuilder {
            adapter,
            features: wgpu::Features::default(),
            limits: wgpu::Limits::default(),
        }
    }

    pub async fn build(self) -> Result<(Device, Queue), Error> {
        let (device, queue) = self
            .adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: self.features,
                    required_limits: self.limits,
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None, // Trace path
            )
            .await?;
        let device = Device::new(self.adapter, device);
        let queue = Queue::new(queue);
        Ok((device, queue))
    }

    pub fn with_features(mut self, features: wgpu::Features) -> Self {
        self.features |= features;
        self
    }

    pub fn with_limits(mut self, limits: wgpu::Limits) -> Self {
        self.limits = limits;
        self
    }

    pub fn with_webgl_limits(mut self) -> Self {
        self.limits = wgpu::Limits::downlevel_webgl2_defaults();
        self
    }
}
