use std::ops::Deref;

use crate::{Adapter, Device, Error, Result};

pub struct Surface<'window> {
    surface: wgpu::Surface<'window>,
}

impl<'window> Surface<'window> {
    pub fn new(surface: wgpu::Surface<'window>) -> Self {
        Self { surface }
    }

    pub fn with_size(self, width: u32, height: u32) -> ConfigBuilder<'window> {
        assert!(width > 0, "width should be > 0");
        assert!(height > 0, "height should be > 0");
        ConfigBuilder {
            surface: self.surface,
            config: wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: wgpu::TextureFormat::Rgba8Unorm,
                width,
                height,
                present_mode: wgpu::PresentMode::default(),
                alpha_mode: wgpu::CompositeAlphaMode::default(),
                view_formats: Vec::new(),
                desired_maximum_frame_latency: 2,
            },
        }
    }
}

impl<'window> Deref for Surface<'window> {
    type Target = wgpu::Surface<'window>;
    fn deref(&self) -> &Self::Target {
        &self.surface
    }
}

// Surface builder implementation.

pub struct ConfigBuilder<'window> {
    surface: wgpu::Surface<'window>,
    config: wgpu::SurfaceConfiguration,
}

impl<'window> ConfigBuilder<'window> {
    pub fn configure(mut self, adapter: &Adapter) -> Result<Self> {
        let surface_caps = self.surface.get_capabilities(adapter);
        self.config.format = surface_caps.formats.first().copied().ok_or(Error::CreateAdapterError)?;
        self.config.alpha_mode = surface_caps
            .alpha_modes
            .first()
            .copied()
            .ok_or(Error::CreateAdapterError)?;
        Ok(self)
    }

    pub fn bind_device(self, device: &Device) -> BoundSurface<'window> {
        self.surface.configure(device, &self.config);
        BoundSurface {
            surface: self.surface,
            config: self.config,
        }
    }

    pub fn with_presentation_mode(mut self, mode: wgpu::PresentMode) -> Self {
        self.config.present_mode = mode;
        self
    }

    pub fn with_maximum_frame_latency(mut self, latency: u32) -> Self {
        self.config.desired_maximum_frame_latency = latency;
        self
    }
}

// Bound surface implementation.

pub struct BoundSurface<'window> {
    surface: wgpu::Surface<'window>,
    config: wgpu::SurfaceConfiguration,
}

impl<'window> BoundSurface<'window> {
    pub fn resize(&mut self, device: &Device, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(device, &self.config);
    }
}

impl<'window> Deref for BoundSurface<'window> {
    type Target = wgpu::Surface<'window>;
    fn deref(&self) -> &Self::Target {
        &self.surface
    }
}
