//! This module provides the implementation of the `Tensor` struct, which represents a tensor on the WGPU backend.
//! It includes functionality for creating tensors, managing their data, and interfacing with the GPU for compute operations.

use std::borrow::Cow;
use std::cell::OnceCell;
use std::marker::PhantomData;
use std::rc::Rc;

use async_trait::async_trait;
use tengu_backend::Error;
use tengu_tensor::StorageType;
use tengu_tensor::Tensor as RawTensor;
use tengu_utils::Label;
use tengu_wgpu::{Buffer, BufferUsage, ByteSize, Encoder};

use crate::source::Source;
use crate::Backend as WGPUBackend;

/// Represents a tensor on the WGPU backend.
pub struct Tensor<T> {
    backend: Rc<WGPUBackend>,
    label: Label,
    count: usize,
    shape: Vec<usize>,
    staging_buffer: OnceCell<Buffer>,
    buffer: Rc<Buffer>,
    phantom: PhantomData<T>,
}

impl<T: StorageType> Tensor<T> {
    /// Creates a new `Tensor` with the specified backend, label, shape, and buffer.
    ///
    /// # Parameters
    /// - `backend`: A reference-counted pointer to the WGPU backend.
    /// - `label`: A string label for identifying the tensor.
    /// - `shape`: The shape of the tensor as a vector of unsigned integers.
    /// - `buffer`: The GPU buffer storing the tensor's data.
    ///
    /// # Returns
    /// A new instance of `Tensor`.
    pub fn new(
        backend: &Rc<WGPUBackend>,
        label: impl Into<Label>,
        shape: impl Into<Vec<usize>>,
        buffer: Buffer,
    ) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            backend: Rc::clone(backend),
            label: label.into(),
            count,
            shape,
            staging_buffer: OnceCell::new(),
            buffer: buffer.into(),
            phantom: PhantomData,
        }
    }

    /// Returns a reference to the tensor's staging object, initializing it if necessary.
    ///
    /// # Returns
    /// A reference to the tensor's staging object.
    fn stage(&self) -> &Buffer {
        self.staging_buffer.get_or_init(|| {
            let size = self.count.of::<T>();
            self.backend
                .device()
                .buffer::<T>(self.label.value(), BufferUsage::Staging)
                .empty(size)
        })
    }
}

// NOTE: Source trait implementation.

#[async_trait]
impl<T: StorageType> Source for Tensor<T> {
    /// Returns the label of the tensor.
    ///
    /// # Returns
    /// The label of the tensor.
    fn label(&self) -> &str {
        self.label.value()
    }

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
        encoder.copy_buffer(&self.buffer, self.stage());
    }
}

// NOTE: Tensor trait implementation.

impl<T: StorageType> RawTensor<T> for Tensor<T> {
    /// Returns the label of the tensor.
    ///
    /// # Returns
    /// The label of the tensor.
    fn label(&self) -> &str {
        self.label.value()
    }

    /// Returns the number of elements in the tensor.
    ///
    /// # Returns
    /// The number of elements in the tensor.
    fn count(&self) -> usize {
        self.count
    }

    /// Returns the shape of the tensor.
    ///
    /// # Returns
    /// The shape of the tensor as a slice of unsigned integers.
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Retrieves staging buffer data from the GPU memory into the CPU buffer.
    ///
    /// # Returns
    /// A `Cow` containing either a reference or owned buffer with the tensor data.
    async fn retrieve(&self) -> anyhow::Result<Cow<'_, [T::IOType]>> {
        let staging_buffer = self.stage();
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.backend.device().poll(wgpu::Maintain::wait()).panic_on_timeout();
        receiver
            .recv_async()
            .await
            .map_err(|e| Error::WGPUError(e.into()))?
            .map_err(|e| Error::WGPUError(e.into()))?;
        let data = buffer_slice.get_mapped_range();
        let buffer = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        Ok(buffer.into())
    }
}

#[cfg(test)]
mod tests {
    use crate::source::Source;
    use crate::Backend as WGPUBackend;
    use pretty_assertions::assert_eq;
    use tengu_backend::Backend;

    #[tokio::test]
    async fn tensor_emit() {
        let backend = WGPUBackend::new().await.unwrap();
        let tensor = backend.zero::<u32>("tenzor", [6]);
        assert_eq!(tensor.label(), "tenzor");
    }
}
