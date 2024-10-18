//! This module defines the `Linker` trait, which is used for propagating tensor data between different parts
//! of a computation graph. The trait provides an interface for copying tensor data on a specified backend.

use crate::{Backend, StorageType};

/// The `Linker` trait defines a set of operations for propagating tensor data within a computation graph.
/// Types that implement this trait can manage the copying of tensor data between different storage
/// locations or parts of the computation graph.
pub trait Linker<'a> {
    /// The type of the backend that this linker interacts with.
    type Backend: Backend;

    /// Copies tensor data from one tensor to another.
    ///
    /// # Parameters
    /// - `from`: A reference to the source tensor from which data will be copied.
    /// - `to`: A reference to the destination tensor to which data will be copied.
    ///
    /// # Type Parameters
    /// - `T`: The type of data stored in the tensors, which must implement the `StorageType` trait.
    fn copy_link<T: StorageType>(
        &mut self,
        from: &<Self::Backend as Backend>::Tensor<T>,
        to: &<Self::Backend as Backend>::Tensor<T>,
    );
}
