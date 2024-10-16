//! This module provides functionality for creating and managing surfaces in the WGPU backend.
//! Surfaces in WGPU are responsible for presenting rendered images to the screen and need to be bound to an adapter to interface with
//! the GPU.
//!
//! ## Surfaces in WGPU
//!
//! In WGPU, a surface represents a platform-specific render target, such as a window or a canvas, where the GPU can present its output.
//! Before a surface can be used, it needs to be bound to an adapter, which provides the necessary capabilities and configurations for
//! rendering.
//!
//! This module includes the `Surface` struct for representing a WGPU surface, the `ConfigBuilder` struct for configuring the surface,
//! and the `BoundSurface` struct for managing a configured surface.
//!
//! ## Module Structs and Methods
//!
//! - `Surface`: Represents a WGPU surface.
//!   - `Surface::new`: Creates a new `Surface` from a WGPU surface.
//!   - `Surface::with_size`: Creates a `ConfigBuilder` for configuring the surface with the specified width and height.
//!
//! - `ConfigBuilder`: Provides a builder pattern for configuring a surface.
//!   - `ConfigBuilder::configure`: Configures the surface with the capabilities of the specified adapter.
//!   - `ConfigBuilder::bind_device`: Binds the configured surface to a device, creating a `BoundSurface`.
//!   - `ConfigBuilder::with_presentation_mode`: Sets the presentation mode for the surface.
//!   - `ConfigBuilder::with_maximum_frame_latency`: Sets the maximum frame latency for the surface.
//!
//! - `BoundSurface`: Represents a surface that has been configured and bound to a device.
//!   - `BoundSurface::resize`: Resizes the surface to the specified width and height.

use std::ops::Deref;

use tracing::trace;

use crate::{Adapter, Device, Error, Result};

// NOTE: Surface implementation.

/// Represents a surface used for rendering in the WGPU backend.
pub struct Surface<'window> {
    surface: wgpu::Surface<'window>,
}

impl<'window> Surface<'window> {
    /// Creates a new `Surface` instance.
    ///
    /// # Parameters
    /// - `surface`: The underlying WGPU surface.
    ///
    /// # Returns
    /// A new `Surface` instance.
    pub fn new(surface: wgpu::Surface<'window>) -> Self {
        Self { surface }
    }

    /// Configures the surface with the specified width and height.
    ///
    /// # Parameters
    /// - `width`: The width of the surface.
    /// - `height`: The height of the surface.
    ///
    /// # Returns
    /// A `ConfigBuilder` for further configuration.
    ///
    /// # Panics
    /// Panics if `width` or `height` is zero.
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

// NOTE: Surface builder implementation.

/// Builder for configuring a surface before it is bound to a device.
pub struct ConfigBuilder<'window> {
    surface: wgpu::Surface<'window>,
    config: wgpu::SurfaceConfiguration,
}

impl<'window> ConfigBuilder<'window> {
    /// Configures the surface with capabilities from the specified adapter.
    ///
    /// # Parameters
    /// - `adapter`: The adapter to get capabilities from.
    ///
    /// # Returns
    /// A `Result` containing the configured `ConfigBuilder` or an error.
    pub fn configure(mut self, adapter: &Adapter) -> Result<Self> {
        let surface_caps = self.surface.get_capabilities(adapter);
        self.config.format = surface_caps.formats.first().copied().ok_or(Error::CreateAdapterError)?;
        self.config.alpha_mode = surface_caps
            .alpha_modes
            .first()
            .copied()
            .ok_or(Error::CreateAdapterError)?;
        trace!(
            "Surface format: {:?}, alpha mode: {:?}",
            self.config.format,
            self.config.alpha_mode
        );
        Ok(self)
    }

    /// Binds the surface to the specified device.
    ///
    /// # Parameters
    /// - `device`: The device to bind the surface to.
    ///
    /// # Returns
    /// A `BoundSurface` instance.
    pub fn bind_device(self, device: &Device) -> BoundSurface<'window> {
        self.surface.configure(device, &self.config);
        trace!("Bound surface to the device");
        BoundSurface {
            surface: self.surface,
            config: self.config,
        }
    }

    /// Sets the presentation mode for the surface.
    ///
    /// # Parameters
    /// - `mode`: The presentation mode to set.
    ///
    /// # Returns
    /// The updated `ConfigBuilder`.
    pub fn with_presentation_mode(mut self, mode: wgpu::PresentMode) -> Self {
        self.config.present_mode = mode;
        trace!("Presentation mode set to {:?}", mode);
        self
    }

    /// Sets the maximum frame latency for the surface.
    ///
    /// # Parameters
    /// - `latency`: The maximum frame latency to set.
    ///
    /// # Returns
    /// The updated `ConfigBuilder`.
    pub fn with_maximum_frame_latency(mut self, latency: u32) -> Self {
        self.config.desired_maximum_frame_latency = latency;
        trace!("Frame latency set to {:?}", latency);
        self
    }
}

// NOTE: Bound surface implementation.

/// Represents a surface that has been bound to a device.
pub struct BoundSurface<'window> {
    surface: wgpu::Surface<'window>,
    config: wgpu::SurfaceConfiguration,
}

impl<'window> BoundSurface<'window> {
    /// Resizes the bound surface.
    ///
    /// # Parameters
    /// - `device`: The device the surface is bound to.
    /// - `width`: The new width of the surface.
    /// - `height`: The new height of the surface.
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
