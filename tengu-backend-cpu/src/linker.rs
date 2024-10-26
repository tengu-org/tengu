//! This module defines the `Linker` struct which implements the `Linker` trait from the `tengu_backend` crate.
//! The `Linker` is responsible for copying data between CPU buffers using the CPU backend.

use tengu_backend::Linker as RawLinker;
use tengu_tensor::StorageType;

use crate::tensor::Tensor;
use crate::Backend as CPUBackend;

/// The `Linker` struct is used to manage and perform copy operations between CPU buffers.
pub struct Linker;

impl RawLinker<CPUBackend> for Linker {
    /// Copies data from one tensor buffer to another. This operation works only on typed tensors,
    /// so the downcast conversion from sources should be performed elsewhere.
    ///
    /// # Parameters
    /// - `from`: A reference to the source tensor.
    /// - `to`: A reference to the destination tensor.
    fn copy_link<T: StorageType>(&mut self, from: &Tensor<T>, to: &Tensor<T>) {
        to.copy_from(from)
    }
}
