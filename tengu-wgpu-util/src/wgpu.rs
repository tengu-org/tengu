use crate::{Device, Error, Queue, Surface};

// Adapter builder implementation.

pub struct AdapterBuilder<'surface, 'window> {
    instance: wgpu::Instance,
    request_adapter_options: wgpu::RequestAdapterOptions<'surface, 'window>,
}

impl<'surface, 'window> AdapterBuilder<'surface, 'window> {
    pub fn new(instance: wgpu::Instance) -> Self {
        Self {
            instance,
            request_adapter_options: wgpu::RequestAdapterOptions::default(),
        }
    }

    pub fn with_surface(mut self, surface: &'surface Surface<'window>) -> Self {
        self.request_adapter_options.compatible_surface = Some(surface);
        self
    }

    pub async fn with_default_adapter(self) -> Result<DeviceBuilder, Error> {
        let adapter = self
            .instance
            .request_adapter(&self.request_adapter_options)
            .await
            .ok_or(Error::CreateAdapterError)?;
        Ok(DeviceBuilder {
            adapter,
            features: wgpu::Features::default(),
            limits: wgpu::Limits::default(),
        })
    }
}

// Device builder implementation.

pub struct DeviceBuilder {
    adapter: wgpu::Adapter,
    features: wgpu::Features,
    limits: wgpu::Limits,
}

impl DeviceBuilder {
    pub async fn request_device(self) -> Result<(Device, Queue), Error> {
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
