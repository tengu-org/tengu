//! This module defines the `UnaryFn` struct and associated functionality for handling unary functions
//! such as logarithm and exponentiation in tensor expressions. This is a helper struct for storing
//! `UnaryFn` variant on the `Expression` struct.

use tengu_backend::Backend;
use tengu_tensor::{Function, StorageType};

use super::Expression;
use crate::node::Node;

// NOTE: UnaryFn

/// Struct representing a unary function applied to a tensor expression.
pub struct UnaryFn<T, B> {
    function: Function,
    expression: Box<dyn Node<T, B>>,
}

impl<T, B> UnaryFn<T, B>
where
    B: Backend + 'static,
    T: StorageType,
{
    /// Creates a new `UnaryFn` instance.
    ///
    /// # Parameters
    /// - `function`: The unary function to apply.
    /// - `expr`: The tensor expression to which the function is applied.
    ///
    /// # Returns
    /// A new `UnaryFn` instance.
    pub fn new<S: StorageType>(function: Function, expr: Expression<T, S, B>) -> Self {
        Self {
            function,
            expression: Box::new(expr),
        }
    }

    /// Creates a new `UnaryFn` instance for the exponentiation function.
    ///
    /// # Parameters
    /// - `expr`: The tensor expression to which the exponentiation function is applied.
    ///
    /// # Returns
    /// A new `UnaryFn` instance with the exponentiation function.
    pub fn exp<S: StorageType>(expr: Expression<T, S, B>) -> Self {
        Self::new(Function::Exp, expr)
    }

    /// Creates a new `UnaryFn` instance for the logarithm function.
    ///
    /// # Parameters
    /// - `expr`: The tensor expression to which the logarithm function is applied.
    ///
    /// # Returns
    /// A new `UnaryFn` instance with the logarithm function.
    pub fn log<S: StorageType>(expr: Expression<T, S, B>) -> Self {
        Self::new(Function::Log, expr)
    }
}

// NOTE: Node implementation.

impl<T, B> Node<T, B> for UnaryFn<T, B>
where
    B: Backend + 'static,
    T: StorageType,
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

    /// Returns a boxed clone of the `UnaryFn` instance.
    ///
    /// # Returns
    /// A boxed clone of the `UnaryFn` instance.
    fn clone_box(&self) -> Box<dyn Node<T, B>> {
        Box::new(self.clone())
    }
}

// NOTE: Clone implementation.

impl<T, B> Clone for UnaryFn<T, B>
where
    B: Backend + 'static,
    T: StorageType,
{
    /// Creates a clone of the `UnaryFn` instance.
    ///
    /// # Returns
    /// A clone of the `UnaryFn` instance.
    fn clone(&self) -> Self {
        Self {
            function: self.function,
            expression: self.expression.clone_box(),
        }
    }
}
