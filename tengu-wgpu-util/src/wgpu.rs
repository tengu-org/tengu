use bon::bon;

use crate::{adapter::AdapterBuilder, Result, Surface};

pub struct WGPU {
    instance: wgpu::Instance,
}

#[bon]
impl WGPU {
    #[builder]
    pub fn new(backends: wgpu::Backends) -> WGPU {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        WGPU { instance }
    }

    pub fn create_surface<'window>(&self, target: impl Into<wgpu::SurfaceTarget<'window>>) -> Result<Surface<'window>> {
        let surface = self.instance.create_surface(target)?;
        Ok(Surface::new(surface))
    }

    pub fn adapter<'surface, 'window>(self) -> AdapterBuilder<'surface, 'window> {
        AdapterBuilder::new(self.instance)
    }
}
