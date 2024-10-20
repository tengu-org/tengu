//! This module defines the `Linker` struct which implements the `Linker` trait from the `tengu_backend` crate.
//! The `Linker` is responsible for copying data between GPU buffers using the WGPU backend.

use tengu_backend::Backend;
use tengu_tensor_traits::StorageType;
use tengu_wgpu::Encoder;

use crate::source::Source;
use crate::Backend as WGPUBackend;

/// The `Linker` struct is used to manage and perform copy operations between GPU buffers.
/// It holds a mutable reference to an `Encoder` which is used to encode the copy operations.
pub struct Linker<'a> {
    encoder: &'a mut Encoder,
}

impl<'a> Linker<'a> {
    /// Creates a new `Linker` instance.
    ///
    /// # Parameters
    /// - `encoder`: A mutable reference to an `Encoder` wrapper from `tengu_wgpu` used for the copy operations.
    ///
    /// # Returns
    /// A new instance of `Linker`.
    pub fn new(encoder: &'a mut Encoder) -> Self {
        Self { encoder }
    }
}

impl<'a> tengu_backend::Linker for Linker<'a> {
    type Backend = WGPUBackend;

    /// Copies data from one tensor buffer to another. This operation works only on typed tensors,
    /// so the downcast conversion from sources should be performed elsewhere.
    ///
    /// # Parameters
    /// - `from`: A reference to the source tensor.
    /// - `to`: A reference to the destination tensor.
    fn copy_link<T: StorageType>(
        &mut self,
        from: &<Self::Backend as Backend>::Tensor<T>,
        to: &<Self::Backend as Backend>::Tensor<T>,
    ) {
        self.encoder.copy_buffer(from.buffer(), to.buffer());
    }
}
