//! This module provides the implementation of the `Tensor` struct, which represents a tensor in the WGPU backend.
//! It includes functionality for creating tensors, managing their data, and interfacing with the GPU for compute operations.
//!
//! ## Overview
//!
//! A `Tensor` is a fundamental data structure used in GPU computations, representing a multi-dimensional array of data.
//! This module defines the `Tensor` struct and implements various traits to integrate tensors with the Tengu backend and WGPU
//! operations.

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use tengu_backend::{Result, StorageType};
use tengu_wgpu::{Buffer, BufferUsage, ByteSize, Encoder};

use crate::probe::Probe;
use crate::source::Source;
use crate::stage::Stage;
use crate::Backend as WGPUBackend;

/// Represents a tensor in the WGPU backend.
pub struct Tensor<T: StorageType> {
    backend: Arc<WGPUBackend>,
    label: String,
    count: usize,
    stage: OnceLock<Stage<T::IOType>>,
    buffer: Arc<Buffer>,
}

impl<T: StorageType> Tensor<T> {
    /// Creates a new `Tensor` with the specified backend, label, element count, and buffer.
    ///
    /// # Parameters
    /// - `backend`: A reference-counted pointer to the WGPU backend.
    /// - `label`: A string label for identifying the tensor.
    /// - `count`: The number of elements in the tensor.
    /// - `buffer`: The GPU buffer storing the tensor's data.
    ///
    /// # Returns
    /// A new instance of `Tensor`.
    pub fn new(backend: &Arc<WGPUBackend>, label: String, count: usize, buffer: Buffer) -> Self {
        Self {
            backend: Arc::clone(backend),
            label,
            count,
            stage: OnceLock::new(),
            buffer: buffer.into(),
        }
    }

    /// Returns a reference to the tensor's staging object, initializing it if necessary.
    ///
    /// # Returns
    /// A reference to the tensor's staging object.
    fn stage(&self) -> &Stage<T::IOType> {
        self.stage.get_or_init(|| {
            let size = self.count.of::<T>();
            let buffer = self
                .backend
                .device()
                .buffer::<T>(self.label(), BufferUsage::Staging)
                .empty(size);
            Stage::new(&self.backend, buffer)
        })
    }
}

// NOTE: Source implementation.

#[async_trait]
impl<T: StorageType> Source for Tensor<T> {
    /// Returns a reference to the GPU buffer storing the tensor's data.
    ///
    /// # Returns
    /// A reference to the `Buffer`.
    fn buffer(&self) -> &Buffer {
        &self.buffer
    }
    /// Copies the tensor's data from the GPU buffer to the staging buffer using the provided encoder.
    ///
    /// # Parameters
    /// - `encoder`: The encoder used to copy the buffer data.
    fn readout(&self, encoder: &mut Encoder) {
        if let Some(stage) = self.stage.get() {
            encoder.copy_buffer(&self.buffer, stage.buffer());
        }
    }

    /// Retrieves staging buffer data from the GPU to CPU buffer.
    ///
    /// # Returns
    /// A `Result` indicating success or failure of the retrieval operation.
    async fn retrieve(&self) -> Result<()> {
        if let Some(stage) = &self.stage.get() {
            stage.retrieve().await?;
        }
        Ok(())
    }

    /// Returns the number of elements in the tensor.
    ///
    /// # Returns
    /// The number of elements in the tensor.
    fn count(&self) -> usize {
        self.count
    }

    /// Returns the label of the tensor.
    ///
    /// # Returns
    /// The label of the tensor.
    fn label(&self) -> &str {
        &self.label
    }
}

// NOTE: Raw tensor implementation.

impl<T: StorageType> tengu_backend::Tensor<T> for Tensor<T> {
    type Probe = Probe<T::IOType>;

    /// Returns the label of the tensor.
    ///
    /// # Returns
    /// The label of the tensor.
    fn label(&self) -> &str {
        &self.label
    }

    /// Returns a reference to the tensor's probe, initializing it if necessary.
    ///
    /// # Returns
    /// A reference to the tensor's probe.
    fn probe(&self) -> Self::Probe {
        Probe::new(self.stage().receiver())
    }
}

#[cfg(test)]
mod tests {
    use crate::Backend as WGPUBackend;
    use pretty_assertions::assert_eq;
    use tengu_backend::{Backend, Tensor};

    #[tokio::test]
    async fn tensor_emit() {
        let backend = WGPUBackend::new().await.unwrap();
        let tensor = backend.zero::<u32>("tenzor", 6);
        assert_eq!(tensor.label(), "tenzor");
    }
}
