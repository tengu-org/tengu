//! This module provides functionality for creating and managing devices in the WGPU backend.
//! Devices in WGPU represent logical connections to the GPU and are responsible for resource creation, command encoding, and submission.
//!
//! ## Devices in WGPU
//!
//! In WGPU, a device is a crucial component that manages GPU resources and operations. It is created from an adapter and is used to
//! create buffers, textures, pipelines, and other GPU resources. A device also provides methods for encoding and submitting commands
//! to the GPU.
//!
//! This module includes the `Device` struct for representing a WGPU device and the `DeviceBuilder` struct for building and requesting
//! devices with specific features and limits.
//!
//! ## Module Structs and Methods
//!
//! - `Device`: Represents a WGPU device and provides methods to create and manage GPU resources.
//!   - `Device::new`: Creates a new `Device` from a WGPU device and queue.
//!   - `Device::encoder`: Creates an encoder for recording GPU commands.
//!   - `Device::compute`: Encodes compute commands using a provided closure.
//!   - `Device::buffer`: Creates a buffer builder for creating GPU buffers.
//!   - `Device::shader`: Creates a shader module from WGSL source code.
//!   - `Device::submit`: Submits a command buffer to the GPU queue for execution.
//!   - `Device::layout`: Creates a layout builder for creating bind group layouts and pipelines.
//!
//! - `DeviceBuilder`: Provides a builder pattern for requesting devices from a WGPU adapter.
//!   - `DeviceBuilder::new`: Creates a new `DeviceBuilder` for the specified adapter.
//!   - `DeviceBuilder::request`: Requests a device asynchronously and returns a `Device` if successful.
//!   - `DeviceBuilder::with_features`: Sets the required features for the device.
//!   - `DeviceBuilder::with_limits`: Sets the required limits for the device.
//!   - `DeviceBuilder::with_webgl_limits`: Sets the WebGL-compatible limits for the device.

use std::ops::Deref;

use crate::buffer::BufferBuilder;
use crate::pipeline::LayoutBuilder;
use crate::{BufferUsage, Encoder, Error};

/// Represents a WGPU device and its associated queue.
pub struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Device {
    /// Creates a new `Device` instance.
    ///
    /// # Parameters
    /// - `device`: The WGPU device.
    /// - `queue`: The queue associated with the device.
    ///
    /// # Returns
    /// A new `Device` instance.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue) -> Device {
        Self { device, queue }
    }

    /// Creates a new command encoder with the specified label.
    ///
    /// # Parameters
    /// - `label`: A label for the command encoder.
    ///
    /// # Returns
    /// An `Encoder` instance.
    pub fn encoder(&self, label: &str) -> Encoder {
        Encoder::new(self, label)
    }

    /// Executes a series of commands encapsulated in a closure and returns the resulting command buffer.
    ///
    /// # Parameters
    /// - `label`: A label for the command encoder.
    /// - `commands`: A closure that takes a mutable reference to an `Encoder` and encodes commands using the provided
    ///   command buffer.
    ///
    /// # Returns
    /// A `wgpu::CommandBuffer` containing the encoded commands.
    pub fn compute(&self, label: &str, commands: impl FnOnce(&mut Encoder)) -> wgpu::CommandBuffer {
        let mut encoder = Encoder::new(self, label);
        commands(&mut encoder);
        encoder.finish()
    }

    /// Creates a new buffer builder for the specified buffer usage.
    ///
    /// # Parameters
    /// - `buffer_kind`: The usage type of the buffer.
    ///
    /// # Returns
    /// A `BufferBuilder` instance.
    pub fn buffer<T>(&self, buffer_kind: BufferUsage) -> BufferBuilder {
        BufferBuilder::new(self, buffer_kind)
    }

    /// Creates a new shader module from the specified WGSL source code.
    ///
    /// # Parameters
    /// - `label`: A label for the shader module.
    /// - `source`: The WGSL source code for the shader.
    ///
    /// # Returns
    /// A `wgpu::ShaderModule` instance.
    pub fn shader(&self, label: &str, source: &str) -> wgpu::ShaderModule {
        self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }

    /// Submits a command buffer to the queue for execution.
    ///
    /// # Parameters
    /// - `commands`: The command buffer to submit.
    pub fn submit(&self, commands: wgpu::CommandBuffer) {
        self.queue.submit(std::iter::once(commands));
    }

    /// Creates a new layout builder for configuring pipeline layouts.
    ///
    /// # Returns
    /// A `LayoutBuilder` instance.
    pub fn layout(&self) -> LayoutBuilder {
        LayoutBuilder::new(self)
    }

    /// Returns the limits of the device.
    ///
    /// # Returns
    /// The device limits.
    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }
}

impl Deref for Device {
    type Target = wgpu::Device;
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

// NOTE: Device builder implementation

/// Builder for creating and configuring a WGPU device.
pub struct DeviceBuilder {
    adapter: wgpu::Adapter,
    features: wgpu::Features,
    limits: wgpu::Limits,
}

impl DeviceBuilder {
    /// Creates a new `DeviceBuilder` instance.
    ///
    /// # Parameters
    /// - `adapter`: The adapter to use for creating the device.
    ///
    /// # Returns
    /// A new `DeviceBuilder` instance.
    pub fn new(adapter: wgpu::Adapter) -> Self {
        let max_storage_buffers_per_shader_stage = adapter.limits().max_storage_buffers_per_shader_stage;
        DeviceBuilder {
            adapter,
            features: wgpu::Features::default(),
            limits: wgpu::Limits {
                max_storage_buffers_per_shader_stage,
                ..Default::default()
            },
        }
    }

    /// Requests a new `Device` instance asynchronously.
    ///
    /// # Returns
    /// A `Result` containing the created `Device` or an error.
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

    /// Adds additional features to the device.
    ///
    /// # Parameters
    /// - `features`: The features to add.
    ///
    /// # Returns
    /// The updated `DeviceBuilder`.
    pub fn with_features(mut self, features: wgpu::Features) -> Self {
        self.features |= features;
        self
    }

    /// Sets the resource limits for the device.
    ///
    /// # Parameters
    /// - `limits`: The limits to set.
    ///
    /// # Returns
    /// The updated `DeviceBuilder`.
    pub fn with_limits(mut self, limits: wgpu::Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Sets the resource limits for the device to WebGL2 defaults.
    ///
    /// # Returns
    /// The updated `DeviceBuilder`.
    pub fn with_webgl_limits(mut self) -> Self {
        self.limits = wgpu::Limits::downlevel_webgl2_defaults();
        self
    }
}
