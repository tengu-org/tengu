//! This module defines the `Tensor` trait which represents a tensor in the Tengu backend.
//! Tensors are fundamental data structures used in numerical computations and are typically
//! stored on the GPU. This trait outlines the necessary operations and associated types
//! required for a particular implemntation on some backend.

#![allow(async_fn_in_trait)]

use std::borrow::Cow;

pub use arithmetic::Arithmetic;
pub use cast::Cast;
pub use copy_from::CopyFrom;
pub use relational::Relational;
pub use unary_fn::UnaryFn;

use crate::StorageType;

mod arithmetic;
mod cast;
mod copy_from;
mod relational;
mod unary_fn;

/// A trait for representing tensors in the Tengu backend.
///
/// This trait defines the necessary operations and associated types required for
/// interacting with tensors in a generic and type-safe manner. Implementors of
/// this trait must specify the type of probe that can be bound to the tensor.
pub trait Tensor<T: StorageType>: Clone {
    /// Returns the label of the tensor.
    ///
    /// # Returns
    /// A string slice representing the label of the tensor if this tensor has a label,
    /// `None` otherwise.
    fn label(&self) -> Option<&str>;

    /// Returns the number of elements in the tensor.
    ///
    /// # Returns
    /// The number of elements in the tensor.
    fn count(&self) -> usize;

    /// Returns the shape of the tensor.
    ///
    /// # Returns
    /// The shape of the tensor as a slice of unsigned integers.
    fn shape(&self) -> &[usize];

    /// Retrieves the data from the tensor.
    ///
    /// # Returns
    /// A result containing a reference to the data stored in the tensor.
    async fn retrieve(&self) -> anyhow::Result<Cow<'_, [T::IOType]>>;
}
