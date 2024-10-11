//! This module provides an implementation of the WGPU backend for tensor operations. It defines
//! the `WGPU` struct, which serves as the entry point for creating GPU instances, surfaces, and
//! adapters. This backend leverages the wgpu library to interface with various GPU backends.

use bon::bon;

use crate::adapter::AdapterBuilder;
use crate::{Device, Result, Surface};

/// The `WGPU` struct represents an instance of the WGPU backend. It is responsible for creating GPU instances, surfaces, and adapters.
pub struct WGPU {
    instance: wgpu::Instance,
}

#[bon]
impl WGPU {
    /// Creates a new instance of the `WGPU` backend with the specified GPU backends.
    ///
    /// # Parameters
    /// - `backends`: The GPU backends to use (e.g., Vulkan, Metal, DX12, etc.).
    ///
    /// # Returns
    /// A new `WGPU` instance.
    #[builder]
    pub fn new(backends: wgpu::Backends) -> WGPU {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        WGPU { instance }
    }

    /// Creates a new surface for rendering.
    ///
    /// # Parameters
    /// - `target`: The target window or surface to render to.
    ///
    /// # Returns
    /// A `Result` containing the created `Surface` or an error.
    pub fn create_surface<'window>(&self, target: impl Into<wgpu::SurfaceTarget<'window>>) -> Result<Surface<'window>> {
        let surface = self.instance.create_surface(target)?;
        Ok(Surface::new(surface))
    }

    /// Creates a new `AdapterBuilder` for configuring and requesting a GPU adapter.
    ///
    /// # Returns
    /// An `AdapterBuilder` instance.
    pub fn adapter<'surface, 'window>(self) -> AdapterBuilder<'surface, 'window> {
        AdapterBuilder::new(self.instance)
    }

    /// Creates a default GPU context using the primary backend.
    ///
    /// # Returns
    /// A `Result` containing the created `Device` or an error.
    pub async fn default_context() -> Result<Device> {
        let instance = Self::builder().backends(wgpu::Backends::PRIMARY).build();
        instance.adapter().request().await?.device().request().await
    }
}
