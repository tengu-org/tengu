//! This module defines the `Source` trait which is used to represent a data source that can provide
//! a buffer, a label, and a methods to readout and retrieve and  data from the GPU. It is needed to treat tensor in
//! uniform fashion, irrespecitve of their underlying storage type.

use tengu_tensor::StorageType;
use tengu_tensor_wgpu::Tensor;
use tengu_wgpu::{Buffer, Encoder};

/// The `Source` trait represents a "type-less" tensor. It is implemented only by tensor and used
/// by the `Processor` to handle all tensors in a uniform fashion.
pub trait Source {
    /// Returns the label of the source.
    ///
    /// # Returns
    /// A string slice representing the label of the source.
    fn label(&self) -> Option<&str>;

    /// Returns a reference to the buffer associated with the source.
    ///
    /// # Returns
    /// A reference to the tengu-wgpu `Buffer` object.
    fn buffer(&self) -> &Buffer;

    /// Performs a readout operation using the provided encoder.
    ///
    /// # Parameters
    /// - `encoder`: A mutable reference to an `Encoder` object used for the readout operation.
    fn readout(&self, encoder: &mut Encoder);
}

impl<T: StorageType> Source for Tensor<T> {
    fn label(&self) -> Option<&str> {
        <Self as tengu_tensor::Tensor<T>>::label(self)
    }

    fn buffer(&self) -> &Buffer {
        self.buffer()
    }

    fn readout(&self, encoder: &mut Encoder) {
        self.readout(encoder)
    }
}
