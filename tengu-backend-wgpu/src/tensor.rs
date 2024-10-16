//! # Tensor Module
//!
//! This module provides the implementation of the `Tensor` struct, which represents a tensor in the WGPU backend.
//! It includes functionality for creating tensors, managing their data, and interfacing with the GPU for compute operations.
//!
//! ## Overview
//!
//! A `Tensor` is a fundamental data structure used in GPU computations, representing a multi-dimensional array of data.
//! This module defines the `Tensor` struct and implements various traits to integrate tensors with the Tengu backend and WGPU
//! operations.

use std::{cell::OnceCell, rc::Rc};
use tengu_backend::{Backend, StorageType};
use tengu_wgpu::{Buffer, Encoder};

use crate::probe::Probe;
use crate::source::Source;
use crate::Backend as WGPUBackend;

/// Represents a tensor in the WGPU backend.
pub struct Tensor<T: StorageType> {
    backend: Rc<WGPUBackend>,
    label: String,
    count: usize,
    probe: OnceCell<Probe<T::IOType>>,
    buffer: Rc<Buffer>,
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
    pub fn new(backend: &Rc<WGPUBackend>, label: String, count: usize, buffer: Buffer) -> Self {
        Self {
            backend: Rc::clone(backend),
            label,
            count,
            probe: OnceCell::new(),
            buffer: buffer.into(),
        }
    }
}

// NOTE: Source implementation.

impl<T: StorageType> Source for Tensor<T> {
    /// Returns a reference to the GPU buffer storing the tensor's data.
    ///
    /// # Returns
    /// A reference to the `Buffer`.
    fn buffer(&self) -> &Buffer {
        &self.buffer
    }
    /// Copies the tensor's data from the GPU buffer to the probe's buffer using the provided encoder.
    ///
    /// # Parameters
    /// - `encoder`: The encoder used to copy the buffer data.
    fn readout(&self, encoder: &mut Encoder) {
        if let Some(probe) = self.probe.get() {
            encoder.copy_buffer(&self.buffer, probe.buffer());
        }
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
    fn probe(&self) -> &Self::Probe {
        self.probe.get_or_init(|| self.backend.probe::<T>(self.count))
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
