//! This module provides functionality for creating and managing GPU buffers in the WGPU backend.
//! It includes definitions for buffer usage types and builders for buffer creation.
//!
//! ## Buffers in WGPU
//!
//! Buffers are a fundamental part of GPU programming. They are used to store data that can be accessed and manipulated by the GPU.
//! In WGPU, buffers can be used for various purposes such as storing vertex data, uniform data, and storage data.
//! Buffers are allocated on the GPU and can be accessed by shaders for computation.
//!
//! ## Building Buffers in Rust
//!
//! This module provides a `BufferBuilder` which allows for a more natural and idiomatic way to create buffers in Rust.
//! The builder pattern is used to configure and create buffers with ease, allowing for better readability and maintainability of code.
//! Here's an example of how to create a buffer using the builder pattern:
//!
//! ```rust
//! let buffer = BufferBuilder::new(&device, BufferUsage::ReadWrite)
//!     .with_label("MyBuffer")
//!     .with_data(&[1, 2, 3, 4]);
//! ```
//!
//! ## Buffer Usage Variants
//!
//! The `BufferUsage` enum defines different usage types for buffers, corresponding to WGPU-defined usages:
//! - `Staging`: Used for staging data to be transferred to the GPU. Corresponds to `MAP_READ | COPY_DST`. These buffers are typically
//!   used for uploading data to the GPU or reading data back from the GPU.
//! - `Read`: Used for reading data from the GPU. Corresponds to `STORAGE | COPY_SRC`. These buffers are mainly used when the data will
//!   be read by shaders.
//! - `Write`: Used for writing data to the GPU. Corresponds to `STORAGE | COPY_DST`. These buffers are used when the GPU will write
//!   data that the CPU will read.
//! - `ReadWrite`: Used for both reading and writing data to/from the GPU. Corresponds to `STORAGE | COPY_SRC | COPY_DST`. These
//!   buffers offer the most flexibility as they can be used for both read and write operations.
//!
//! ## Staging Buffers
//!
//! Staging buffers are a special type of buffer used for transferring data between the CPU and GPU. They are needed because direct
//! access to GPU memory is often restricted or inefficient. By using staging buffers, data can be efficiently transferred to and from
//! the GPU. This is particularly useful for initializing buffers with data or reading back results from computations.

use std::ops::Deref;
use wgpu::util::DeviceExt;

use crate::Device;

/// Enumerates the different usages for a buffer.
#[derive(Copy, Clone, Debug)]
pub enum BufferUsage {
    /// Buffer used for staging.
    Staging,
    /// Buffer used for reading.
    Read,
    /// Buffer used for both reading and writing.
    ReadWrite,
    /// Buffer used for writing.
    Write,
}

impl BufferUsage {
    /// Returns the corresponding WGPU buffer usage flags.
    ///
    /// # Returns
    /// A `wgpu::BufferUsages` value representing the buffer usage.
    fn usage(&self) -> wgpu::BufferUsages {
        use wgpu::BufferUsages as Usage;
        match self {
            Self::Staging => Usage::MAP_READ | Usage::COPY_DST,
            Self::Read => Usage::STORAGE | Usage::COPY_SRC,
            Self::Write => Usage::STORAGE | Usage::COPY_DST,
            Self::ReadWrite => Usage::STORAGE | Usage::COPY_SRC | Usage::COPY_DST,
        }
    }
}

/// Represents a GPU buffer in the WGPU backend.
pub struct Buffer {
    buffer: wgpu::Buffer,
    usage: BufferUsage,
}

impl Buffer {
    /// Creates a new `Buffer` instance.
    ///
    /// # Parameters
    /// - `buffer`: The WGPU buffer.
    /// - `usage`: The usage type of the buffer.
    ///
    /// # Returns
    /// A new `Buffer` instance.
    fn new(buffer: wgpu::Buffer, usage: BufferUsage) -> Self {
        Self { buffer, usage }
    }

    /// Returns the usage type of the buffer.
    ///
    /// # Returns
    /// A `BufferUsage` value.
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }
}

impl Deref for Buffer {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

// NOTE: BufferBuilder implementation

/// Builder for creating and configuring a GPU buffer.
pub struct BufferBuilder<'a, 'device> {
    device: &'device Device,
    label: Option<&'a str>,
    usage: BufferUsage,
}

impl<'a, 'device> BufferBuilder<'a, 'device> {
    /// Creates a new `BufferBuilder` instance.
    ///
    /// # Parameters
    /// - `device`: The device to use for creating the buffer.
    /// - `usage`: The usage type of the buffer.
    ///
    /// # Returns
    /// A new `BufferBuilder` instance.
    pub fn new(device: &'device Device, usage: BufferUsage) -> Self {
        Self {
            device,
            label: None,
            usage,
        }
    }

    /// Sets the label for the buffer.
    ///
    /// # Parameters
    /// - `label`: The label to set.
    ///
    /// # Returns
    /// The updated `BufferBuilder`.
    pub fn with_label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    /// Creates an empty buffer with the specified size.
    ///
    /// # Parameters
    /// - `size`: The size of the buffer in bytes.
    ///
    /// # Returns
    /// A `Buffer` instance.
    pub fn empty(self, size: usize) -> Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: self.label,
            size: size as u64,
            usage: self.usage.usage(),
            mapped_at_creation: false,
        });
        Buffer::new(buffer, self.usage)
    }

    /// Creates a buffer initialized with the specified data.
    ///
    /// # Parameters
    /// - `data`: A slice of data to initialize the buffer with.
    ///
    /// # Returns
    /// A `Buffer` instance.
    pub fn with_data<T>(self, data: &'a [T]) -> Buffer
    where
        T: bytemuck::Pod,
    {
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: self.label,
            contents: bytemuck::cast_slice(data),
            usage: self.usage.usage(),
        });
        Buffer::new(buffer, self.usage)
    }
}
