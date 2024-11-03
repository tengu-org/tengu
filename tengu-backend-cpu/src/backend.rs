//! This module defines the `Backend` struct which implements the `Backend` trait from the `tengu_backend` crate.
//! The `Backend` struct is responsible for managing creating various backend subsystems like
//! processor and linker. In case of this CPU-based implementation, most of the work is done by the
//! `Processor` instance, so `Compute` and `Readout` operators do no job at all.

use std::rc::Rc;

use tengu_backend::Result;
use tengu_tensor::{IOType, StorageType};
use tengu_tensor_cpu::Tensor;
use tengu_utils::Label;

use crate::limits::Limits;
use crate::operation::{Compute, Propagate, Readout};

pub struct Backend;

/// The `Backend` struct is responsible for tensor creation and various subsystem initialization
/// procedures.
impl tengu_backend::Backend for Backend {
    type Tensor<T: StorageType> = Tensor<T>;
    type Compute = Compute;

    // NOTE: Operations

    type Propagate = Propagate;
    type Readout = Readout;
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
        label: impl Into<Label>,
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
    fn zero<T: StorageType>(self: &Rc<Self>, label: impl Into<Label>, shape: impl Into<Vec<usize>>) -> Self::Tensor<T> {
        Tensor::repeat(label, shape, T::default())
    }
}
