//! This module defines the `Backend` trait, which serves as an interface for tensor computation backends.
//! The trait provides various methods and associated types for creating and manipulating tensors, processors,
//! compute instances, linkers, and readouts within a backend.

#![allow(async_fn_in_trait)]

use std::collections::HashSet;
use std::rc::Rc;

use tengu_backend_tensor::{IOType, StorageType, Tensor};

use crate::*;

/// The `Backend` trait provides an interface for tensor computation backends. It defines various associated
/// types and methods for creating and managing tensors, processors, compute instances, linkers, and readouts.
pub trait Backend: Sized {
    /// The underlying tensor type used by all backend subsystems.
    type Tensor<T: StorageType>: Tensor<T>;

    /// The type of the node processor that will construct computation objects.
    type Processor<'a>: Processor<'a, Self>
    where
        Self: 'a;

    /// The underlying raw compute type.
    type Compute<'a>: Compute<Self>
    where
        Self: 'a;

    /// The underlying linker type.
    type Linker<'a>: Linker<Self>;

    /// The underlying readout type.
    type Readout<'a>: Readout<Self>;

    /// The underliying limits type.
    type Limits: Limits;

    /// Asynchronously creates a new backend instance.
    ///
    /// # Returns
    /// A result wrapping an `Arc` to the new backend instance.
    async fn new() -> Result<Rc<Self>>;

    /// Returns the limits of the backend.
    ///
    /// # Returns
    /// The limits of the backend.
    fn limits(&self) -> Self::Limits;

    /// Creates a processor that will be used to recursively process tensor AST and
    /// convert them to the representation suitable for backend.
    ///
    /// # Returns
    /// A processor instance for the backend.
    fn processor<'a>(&self, readouts: &'a HashSet<String>) -> Self::Processor<'a>;

    /// Propagates buffers through links using the provided callback.
    ///
    /// # Parameters
    /// - `call`: A callback function that takes the linker as an argument propagates the
    ///   information through the link.
    fn propagate(&self, call: impl FnOnce(Self::Linker<'_>));

    /// Updates staging data by copying graph state into the staging buffers.
    ///
    /// # Parameters
    /// - `label`: A label for the readout operation, to be used by backend for debugging purposes.
    /// - `call`: A callback function that takes the readout as an argument and performs the
    ///   staging operation.
    fn readout(&self, label: &str, call: impl FnOnce(Self::Readout<'_>));

    /// Computes the specified function on the backend using the provided callback.
    ///
    /// # Parameters
    /// - `label`: A label for the computation, to be used by backend for debugging purposes.
    /// - `call`: A callback function that takes the compute instance as an argument.
    fn compute<F>(&self, label: &str, call: F) -> Result<()>
    where
        F: FnOnce(Self::Compute<'_>) -> anyhow::Result<()>;

    /// Creates a new zero-initialized tensor with the specified label and element count.
    ///
    /// # Parameters
    /// - `label`: A label for the tensor.
    /// - `shape`: The shape of the tensor.
    ///
    /// # Returns
    /// A new zero-initialized tensor.
    fn zero<T: StorageType>(self: &Rc<Self>, label: impl Into<String>, shape: impl Into<Vec<usize>>)
        -> Self::Tensor<T>;

    /// Creates a new tensor with the specified label and data.
    ///
    /// # Parameters
    /// - `label`: A label for the tensor.
    /// - `shape`: The shape of the tensor.
    /// - `data`: A slice of data to initialize the tensor.
    ///
    /// # Returns
    /// A new tensor initialized with the given data.
    fn tensor<T: IOType>(
        self: &Rc<Self>,
        label: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        data: &[T],
    ) -> Self::Tensor<T>;
}
