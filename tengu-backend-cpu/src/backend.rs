//! This module defines the `Backend` struct which implements the `Backend` trait from the `tengu_backend` crate.
//! The `Backend` struct is responsible for managing creating various backend subsystems like
//! processor and linker. In case of this CPU-based implementation, most of the work is done by the
//! `Processor` instance, so `Compute` and `Readout` operators do no job at all.

use std::collections::HashSet;
use std::rc::Rc;

use tengu_backend::{Error, Result};
use tengu_backend_tensor::{IOType, StorageType};

use crate::compute::Compute;
use crate::limits::Limits;
use crate::linker::Linker;
use crate::processor::Processor;
use crate::readout::Readout;
use crate::tensor::Tensor;

pub struct Backend;

/// The `Backend` struct is responsible for tensor creation and various subsystem initialization
/// procedures.
impl tengu_backend::Backend for Backend {
    type Tensor<T: StorageType> = Tensor<T>;
    type Compute<'a> = Compute;
    type Processor<'a> = Processor<'a>;
    type Linker<'a> = Linker;
    type Readout<'a> = Readout;
    type Limits = Limits;

    /// Creates a new `Backend` instance asynchronously.
    ///
    /// # Returns
    /// A result containing a reference-counted `Backend` instance or an error.
    async fn new() -> Result<Rc<Self>> {
        Ok(Rc::new(Self))
    }

    /// Returns the limits of the backend.
    ///
    /// # Returns
    /// The limits of the backend.
    fn limits(&self) -> Self::Limits {
        Limits
    }

    /// Creates a new `Processor` instance.
    ///
    /// # Parameters
    /// - `readouts`: A set of readout labels to be used by the processor.
    ///
    /// # Returns
    /// A new `Processor` instance.
    fn processor<'a>(&self, readouts: &'a HashSet<String>) -> Self::Processor<'a> {
        Processor::new(readouts)
    }

    /// Propagates data using the provided linker function.
    ///
    /// # Parameters
    /// - `call`: A function that takes a `Linker` and performs data propagation.
    fn propagate(&self, call: impl FnOnce(Self::Linker<'_>)) {
        call(Linker);
    }

    /// Executes a compute pass using the provided compute function. In the case of this CPU
    /// implementation, all the work is done by the processor, so this is essentially a noop.
    ///
    /// # Parameters
    /// - `label`: A label for compute operations.
    /// - `call`: A function that takes a `Compute` and performs compute operations.
    fn compute<F>(&self, _label: &str, call: F) -> Result<()>
    where
        F: FnOnce(Self::Compute<'_>) -> anyhow::Result<()>,
    {
        call(Compute).map_err(Error::ComputeError)
    }

    /// Copies data to staging buffers using the provided staging function. In the case of this CPU
    /// implementation, there is no readout needed since there are no staging buffers on CPU.
    ///
    /// # Parameters
    /// - `label`: A label for the readout operations.
    /// - `call`: A function that takes a `Readout` value and copies data into staging buffers.
    fn readout(&self, _label: &str, call: impl FnOnce(Self::Readout<'_>)) {
        call(Readout);
    }

    /// Creates a new tensor with the provided data.
    ///
    /// # Parameters
    /// - `label`: A label for the tensor.
    /// - `shape`: The shape of the tensor.
    /// - `data`: A slice of data to initialize the tensor with.
    ///
    /// # Returns
    /// A new tensor initialized with the provided data.
    fn tensor<T: IOType>(
        self: &Rc<Self>,
        label: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        data: &[T],
    ) -> Self::Tensor<T> {
        Tensor::new(label, shape, data)
    }

    /// Creates a new zero-initialized tensor with the specified shape.
    ///
    /// # Parameters
    /// - `label`: A label for the tensor.
    /// - `shape`: The shape of the tensor.
    ///
    /// # Returns
    /// A new zero-initialized tensor.
    fn zero<T: StorageType>(
        self: &Rc<Self>,
        label: impl Into<String>,
        shape: impl Into<Vec<usize>>,
    ) -> Self::Tensor<T> {
        Tensor::empty(label, shape)
    }
}
