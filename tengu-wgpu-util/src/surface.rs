use std::ops::Deref;

use crate::{Device, Error, Result};

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
            height,
            width,
            presentation_mode: wgpu::PresentMode::default(),
            maximum_frame_latency: 2,
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
    height: u32,
    width: u32,
    presentation_mode: wgpu::PresentMode,
    maximum_frame_latency: u32,
}

impl<'window> ConfigBuilder<'window> {
    pub fn configure(self, device: &Device) -> Result<BoundSurface<'window>> {
        let surface_caps = self.surface.get_capabilities(device.adapter());
        let format = surface_caps.formats.first().copied().ok_or(Error::CreateAdapterError)?;
        let alpha_mode = surface_caps
            .alpha_modes
            .first()
            .copied()
            .ok_or(Error::CreateAdapterError)?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: self.width,
            height: self.height,
            present_mode: self.presentation_mode,
            alpha_mode,
            view_formats: Vec::new(),
            desired_maximum_frame_latency: self.maximum_frame_latency,
        };
        self.surface.configure(device, &config);
        Ok(BoundSurface {
            surface: self.surface,
            config,
        })
    }

    pub fn with_presentation_mode(mut self, mode: wgpu::PresentMode) -> Self {
        self.presentation_mode = mode;
        self
    }

    pub fn with_maximum_frame_latency(mut self, latency: u32) -> Self {
        self.maximum_frame_latency = latency;
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
