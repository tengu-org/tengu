//! This module provides functionality for creating and managing adapters in the WGPU backend.
//! Adapters in WGPU are responsible for interfacing with the underlying graphics hardware.
//! They provide the capabilities and limits of the GPU, and are used to create devices that execute GPU commands.
//!
//! ## Adapters in WGPU
//!
//! Adapters serve as an abstraction over the physical GPU hardware. They are queried from an instance and provide information
//! about the GPU's features, limits, and supported backends. An adapter is used to create a device, which is then used to
//! create and manage GPU resources such as buffers, textures, and pipelines.
//!
//! This module includes the `Adapter` struct for representing a WGPU adapter and the `AdapterBuilder` struct for building and
//! requesting adapters with specific options.
//!
//! ## Module Structs and Methods
//!
//! - `Adapter`: Represents a WGPU adapter and provides methods to create a device from the adapter.
//!   - `Adapter::new`: Creates a new `Adapter` from a WGPU adapter.
//!   - `Adapter::device`: Creates a `DeviceBuilder` from the adapter for building a GPU device.
//!
//! - `AdapterBuilder`: Provides a builder pattern for requesting adapters from a WGPU instance.
//!   - `AdapterBuilder::new`: Creates a new `AdapterBuilder` for the specified instance.
//!   - `AdapterBuilder::with_surface`: Sets the surface for the adapter to be compatible with.
//!   - `AdapterBuilder::request`: Requests an adapter asynchronously and returns an `Adapter` if successful.

use std::ops::Deref;

use tracing::trace;

use crate::{device::DeviceBuilder, Error, Surface};

// NOTE: Adapter implementation.

/// Represents a GPU adapter in WGPU.
///
/// An adapter provides information about the GPU's capabilities and is used to create devices
/// that manage and execute GPU commands.
pub struct Adapter {
    adapter: wgpu::Adapter,
}

impl Adapter {
    /// Creates a new `Adapter` instance.
    ///
    /// # Parameters
    /// - `adapter`: The WGPU adapter.
    ///
    /// # Returns
    /// A new `Adapter` instance.
    pub fn new(adapter: wgpu::Adapter) -> Self {
        Self { adapter }
    }

    /// Creates a `DeviceBuilder` from the adapter for building a GPU device.
    ///
    /// # Returns
    /// A `DeviceBuilder` instance.
    pub fn device(self) -> DeviceBuilder {
        DeviceBuilder::new(self.adapter)
    }
}

impl Deref for Adapter {
    type Target = wgpu::Adapter;
    fn deref(&self) -> &Self::Target {
        &self.adapter
    }
}

// NOTE: Adapter builder implementation.

/// Builder for creating and configuring a GPU adapter.
///
/// The `AdapterBuilder` allows for specifying options when requesting an adapter, such as
/// setting a compatible surface.
pub struct AdapterBuilder<'surface, 'window> {
    instance: wgpu::Instance,
    request_adapter_options: wgpu::RequestAdapterOptions<'surface, 'window>,
}

impl<'surface, 'window> AdapterBuilder<'surface, 'window> {
    /// Creates a new `AdapterBuilder` instance.
    ///
    /// # Parameters
    /// - `instance`: The WGPU instance to use for requesting the adapter.
    ///
    /// # Returns
    /// A new `AdapterBuilder` instance.
    pub fn new(instance: wgpu::Instance) -> Self {
        Self {
            instance,
            request_adapter_options: wgpu::RequestAdapterOptions::default(),
        }
    }

    /// Sets the surface for the adapter to be compatible with.
    ///
    /// # Parameters
    /// - `surface`: The surface to set.
    ///
    /// # Returns
    /// The updated `AdapterBuilder`.
    pub fn with_surface(mut self, surface: &'surface Surface<'window>) -> Self {
        self.request_adapter_options.compatible_surface = Some(surface);
        self
    }

    /// Requests an adapter asynchronously and returns an `Adapter` if successful.
    ///
    /// # Returns
    /// A `Result` containing an `Adapter` or an `Error` if the adapter creation fails.
    pub async fn request(self) -> Result<Adapter, Error> {
        let adapter = self
            .instance
            .request_adapter(&self.request_adapter_options)
            .await
            .ok_or(Error::CreateAdapterError)?;
        trace!("Requested new adapter");
        Ok(Adapter::new(adapter))
    }
}
