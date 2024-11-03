//! This module defines the `Backend` trait, which serves as an interface for tensor computation backends.
//! The trait provides various methods and associated types for creating and manipulating tensors, processors,
//! compute instances, linkers, and readouts within a backend.

#![allow(async_fn_in_trait)]

use std::rc::Rc;

use tengu_tensor::{IOType, StorageType, Tensor};
use tengu_utils::Label;

use crate::*;

/// The `Backend` trait provides an interface for tensor computation backends. It defines various associated
/// types and methods for creating and managing tensors, processors, compute instances, linkers, and readouts.
pub trait Backend: Sized {
    /// The underlying tensor type used by all backend subsystems.
    type Tensor<T: StorageType>: Tensor<T>;

    /// The underliying limits type.
    type Limits: Limits;

    // NOTE: Operations

    /// The underlying raw compute type.
    type Compute: Compute<Self>;

    /// The underlying propagate operation type.
    type Propagate: Propagate<Self>;

    /// The underlying readout type.
    type Readout: Readout<Self>;

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

    fn operation<O: Operation<Self>>(self: &Rc<Self>, label: impl Into<Label>) -> O {
        O::new(self, label)
    }

    /// Creates a new zero-initialized tensor with the specified label and element count.
    ///
    /// # Parameters
    /// - `label`: A label for the tensor.
    /// - `shape`: The shape of the tensor.
    ///
    /// # Returns
    /// A new zero-initialized tensor.
    fn zero<T: StorageType>(self: &Rc<Self>, label: impl Into<Label>, shape: impl Into<Vec<usize>>) -> Self::Tensor<T>;

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
        label: impl Into<Label>,
        shape: impl Into<Vec<usize>>,
        data: &[T],
    ) -> Self::Tensor<T>;
}
