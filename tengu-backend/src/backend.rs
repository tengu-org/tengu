//! This module defines the `Backend` trait, which serves as an interface for tensor computation backends.
//! The trait provides various methods and associated types for creating and manipulating tensors, processors,
//! compute instances, linkers, and readouts within a backend.

#![allow(async_fn_in_trait)]

use std::rc::Rc;

use crate::linker::Linker;
use crate::readout::Readout;
use crate::{Compute, IOType, Limits, Processor, Result, StorageType, Tensor};

/// The `Backend` trait provides an interface for tensor computation backends. It defines various associated
/// types and methods for creating and managing tensors, processors, compute instances, linkers, and readouts.
pub trait Backend {
    /// The underlying raw tensor type.
    type Tensor<T: StorageType>: Tensor<T>;

    /// The type of the node processor that will construct computation objects.
    type Processor<'a>: Processor<'a, Backend = Self>
    where
        Self: 'a;

    /// The underlying raw compute type.
    type Compute<'a>: Compute<Backend = Self>
    where
        Self: 'a;

    /// The underlying linker type.
    type Linker<'a>: Linker<'a, Backend = Self>;

    /// The underlying readout type.
    type Readout<'a>: Readout<'a, Backend = Self>;

    type Limits: Limits<Backend = Self>;

    /// Asynchronously creates a new backend instance.
    ///
    /// # Returns
    /// A result wrapping an `Rc` to the new backend instance.
    async fn new() -> Result<Rc<Self>>;

    /// Returns the limits of the backend.
    ///
    /// # Returns
    /// The limits of the backend.
    fn limits(&self) -> Self::Limits;

    /// Creates a processor to perform recursive computation of tensor expression ASTs.
    ///
    /// # Returns
    /// A processor instance for the backend.
    fn processor(&self) -> Self::Processor<'_>;

    /// Propagates buffers through links using the provided callback.
    ///
    /// # Parameters
    /// - `call`: A callback function that takes the linker as an argument.
    fn propagate(&self, call: impl FnOnce(Self::Linker<'_>));

    /// Reads out probes using the provided callback.
    ///
    /// # Parameters
    /// - `label`: A label for the readout operation, to be used by backend for debugging purposes.
    /// - `call`: A callback function that takes the readout as an argument.
    fn readout(&self, label: &str, call: impl FnOnce(Self::Readout<'_>));

    /// Computes the specified function on the backend using the provided callback.
    ///
    /// # Parameters
    /// - `label`: A label for the computation, to be used by backend for debugging purposes.
    /// - `call`: A callback function that takes the compute instance as an argument.
    fn compute<F>(&self, label: &str, call: F) -> Result<()>
    where
        F: FnOnce(Self::Compute<'_>) -> Result<()>;

    /// Creates a new zero-initialized tensor with the specified label and element count.
    ///
    /// # Parameters
    /// - `label`: A label for the tensor.
    /// - `count`: Total number of elements in the tensor.
    ///
    /// # Returns
    /// A new zero-initialized tensor.
    fn zero<T: StorageType>(self: &Rc<Self>, label: impl Into<String>, count: usize) -> Self::Tensor<T>;

    /// Creates a new tensor with the specified label and data.
    ///
    /// # Parameters
    /// - `label`: A label for the tensor.
    /// - `data`: A slice of data to initialize the tensor.
    ///
    /// # Returns
    /// A new tensor initialized with the given data.
    fn tensor<T: IOType>(self: &Rc<Self>, label: impl Into<String>, data: &[T]) -> Self::Tensor<T>;

    /// Creates a new probe with the specified label and size.
    ///
    /// # Parameters
    /// - `count`: The number of elements returned by the probe.
    ///
    /// # Returns
    /// A new probe instance.
    fn probe<T: StorageType>(self: &Rc<Self>, count: usize) -> <Self::Tensor<T> as Tensor<T>>::Probe;
}
