//! This module provides functionality for creating and managing command encoders in the WGPU backend.
//! Command encoders are essential in WGPU for recording commands that will be submitted to the GPU for execution.
//! They allow for the batching of multiple operations into a single command buffer, which can then be executed by the GPU.
//!
//! ## Encoders in WGPU
//!
//! In WGPU, command encoders are used to record a variety of GPU commands such as rendering, compute operations, and data transfers.
//! Encoders are created from a device and can record commands to command buffers, which are then submitted to a queue for execution.
//! This mechanism allows for efficient submission of work to the GPU, minimizing CPU-GPU synchronization overhead.
//!
//! ## Compute Passes
//!
//! A compute pass is a sequence of compute operations that are recorded into a command buffer.
//! Compute passes are used to perform general-purpose computations on the GPU using compute shaders.
//! In this module, the `Encoder` struct provides methods to create and manage compute passes, allowing for flexible and efficient GPU
//! computations.
//!
//! ## Module Structs and Methods
//!
//! - `Encoder`: Represents a command encoder in WGPU. It provides methods for creating compute passes, copying buffers, and finalizing
//!   command buffers.
//!   - `new`: Creates a new command encoder with a specified label.
//!   - `pass`: Begins a new compute pass with a specified label and executes a provided closure with the compute pass.
//!   - `readout`: Executes a provided closure with a mutable reference to the encoder.
//!   - `copy_buffer`: Copies data from a source buffer to a destination buffer.
//!   - `finish`: Finalizes the command buffer and returns it for submission to the GPU.

use tracing::trace;

use crate::{Buffer, Device, Error, Result};

/// Represents a command encoder in the WGPU backend.
pub struct Encoder {
    encoder: wgpu::CommandEncoder,
}

impl Encoder {
    /// Creates a new `Encoder` instance.
    ///
    /// # Parameters
    /// - `device`: The device to use for creating the command encoder.
    /// - `label`: A label for the command encoder.
    ///
    /// # Returns
    /// A new `Encoder` instance.
    pub fn new(device: &Device, label: &str) -> Self {
        let encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        trace!("Created command encoder '{label}'");
        Self { encoder }
    }

    /// Begins a compute pass and executes the provided callback.
    ///
    /// # Parameters
    /// - `label`: A label for the compute pass.
    /// - `call`: A callback function to execute within the compute pass.
    ///
    /// # Returns
    /// The updated `Encoder` instance. If an error occurs during the compute pass, it is returned as an `Error`.
    pub fn pass<F>(mut self, label: &str, call: F) -> Result<Self>
    where
        F: FnOnce(wgpu::ComputePass) -> anyhow::Result<()>,
    {
        let compute_pass = self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        trace!("Executing compute pass");
        call(compute_pass).map_err(Error::ComputeError)?;
        Ok(self)
    }

    /// Executes the provided callback for readout operations.
    ///
    /// # Parameters
    /// - `call`: A callback function to execute for readout operations.
    ///
    /// # Returns
    /// The updated `Encoder` instance.
    pub fn readout(mut self, call: impl FnOnce(&mut Encoder)) -> Self {
        trace!("Executing readout");
        call(&mut self);
        self
    }

    /// Copies the contents of one buffer to another.
    ///
    /// # Parameters
    /// - `source`: The source buffer.
    /// - `destination`: The destination buffer.
    pub fn copy_buffer(&mut self, source: &Buffer, destination: &Buffer) {
        let size = source.size();
        self.encoder.copy_buffer_to_buffer(source, 0, destination, 0, size);
    }

    /// Finishes the command encoding and returns the command buffer.
    ///
    /// # Returns
    /// A `wgpu::CommandBuffer` containing the encoded commands.
    pub fn finish(self) -> wgpu::CommandBuffer {
        self.encoder.finish()
    }
}
