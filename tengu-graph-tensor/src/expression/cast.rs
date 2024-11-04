//! This module defines the `Cast` struct and associated functionality for handling type casting in tensor expressions.
//! It leverages the backend processing capabilities to apply type casting on tensor data.

use std::marker::PhantomData;

use tengu_backend::Backend;
use tengu_tensor::StorageType;

use super::Expression;
use crate::node::Node;

/// Struct representing a type cast on a tensor expression.
pub struct Cast<T, S, B> {
    expression: Box<dyn Node<S, B>>,
    phantom: PhantomData<T>,
}

impl<T, S, B> Cast<T, S, B>
where
    B: Backend + 'static,
    T: StorageType,
    S: StorageType,
{
    /// Creates a new `Cast` instance.
    ///
    /// # Parameters
    /// - `expr`: The tensor expression to be cast to a different type.
    ///
    /// # Returns
    /// A new `Cast` instance.
    pub fn new<U: StorageType>(expr: Expression<S, U, B>) -> Self {
        Self {
            expression: Box::new(expr),
            phantom: PhantomData,
        }
    }
}

// NOTE: Node implenentation.

impl<T, S, B> Node<T, B> for Cast<T, S, B>
where
    T: StorageType,
    S: StorageType,
    B: Backend + 'static,
{
    /// Returns the number of elements in the tensor.
    ///
    /// # Returns
    /// The number of elements in the tensor.
    fn count(&self) -> usize {
        self.expression.count()
    }

    /// Returns the shape of the tensor as a slice of dimensions.
    ///
    /// # Returns
    /// A slice representing the dimensions of the tensor.
    fn shape(&self) -> &[usize] {
        self.expression.shape()
    }

    /// Returns a boxed clone of the `Cast` instance.
    ///
    /// # Returns
    /// A boxed clone of the `Cast` instance.
    fn clone_box(&self) -> Box<dyn Node<T, B>> {
        Box::new(self.clone())
    }
}

// NOTE: Clone implementation.

impl<T, S, B> Clone for Cast<T, S, B>
where
    B: Backend + 'static,
    T: StorageType,
    S: StorageType,
{
    /// Creates a clone of the `Cast` instance.
    ///
    /// # Returns
    /// A clone of the `Cast` instance.
    fn clone(&self) -> Self {
        Self {
            expression: self.expression.clone_box(),
            phantom: PhantomData,
        }
    }
}
