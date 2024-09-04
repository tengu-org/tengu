use crate::{wgpu::AdapterBuilder, Result, Surface};

pub struct WGPU {
    instance: wgpu::Instance,
}

impl WGPU {
    pub fn instance() -> InstanceBuilder {
        InstanceBuilder
    }

    pub fn create_surface<'window>(&self, target: impl Into<wgpu::SurfaceTarget<'window>>) -> Result<Surface<'window>> {
        let surface = self.instance.create_surface(target)?;
        Ok(Surface::new(surface))
    }

    pub fn builder<'surface, 'window>(self) -> AdapterBuilder<'surface, 'window> {
        AdapterBuilder::new(self.instance)
    }
}

// Instance builder implementation.

pub struct InstanceBuilder;

impl InstanceBuilder {
    pub fn with_backends(self, backends: wgpu::Backends) -> WGPU {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        WGPU { instance }
    }
}
