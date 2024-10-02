use std::ops::Deref;

use crate::{buffer::BufferBuilder, pipeline::LayoutBuilder, BufferUsage, Encoder, Error};

pub struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Device {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Device {
        Self { device, queue }
    }

    pub fn encoder(&self, label: &str) -> Encoder {
        Encoder::new(self, label)
    }

    pub fn compute<F>(&self, label: &str, commands: F) -> wgpu::CommandBuffer
    where
        F: FnOnce(&mut Encoder),
    {
        let mut encoder = Encoder::new(self, label);
        commands(&mut encoder);
        encoder.finish()
    }

    pub fn buffer<T>(&self, buffer_kind: BufferUsage) -> BufferBuilder {
        BufferBuilder::new(self, buffer_kind)
    }

    pub fn shader(&self, label: &str, source: &str) -> wgpu::ShaderModule {
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }

    pub fn submit(&self, commands: wgpu::CommandBuffer) {
        self.queue.submit(std::iter::once(commands));
    }

    pub fn layout(&self) -> LayoutBuilder {
        LayoutBuilder::new(self)
    }
}

impl Deref for Device {
    type Target = wgpu::Device;
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

// Device builder implementation

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

    pub async fn request(self) -> Result<Device, Error> {
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
        Ok(Device::new(device, queue))
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
