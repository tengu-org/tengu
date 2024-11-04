//! This module defines the `Statement` struct and associated functionality for handling
//! tensor statements within expressions. It ensures that tensor shapes match and provides
//! methods for processing and visiting tensor expressions.

use tengu_backend::Backend;
use tengu_tensor::StorageType;

use super::Expression;
use crate::node::Node;

/// Struct representing a tensor statement, containing an output tensor and an expression tensor.
pub struct Statement<T, B> {
    output: Box<dyn Node<T, B>>,
    expression: Box<dyn Node<T, B>>,
}

impl<T, B> Statement<T, B>
where
    B: Backend + 'static,
    T: StorageType,
{
    /// Creates a new `Statement` instance.
    ///
    /// # Parameters
    /// - `output`: The output tensor expression.
    /// - `expression`: The tensor expression to be assigned to the output.
    ///
    /// # Returns
    /// A new `Statement` instance.
    ///
    /// # Panics
    /// Panics if the shapes of `output` and `expression` do not match.
    pub fn new<S1: StorageType, S2: StorageType>(
        output: Expression<T, S1, B>,
        expression: Expression<T, S2, B>,
    ) -> Self {
        assert_eq!(expression.shape(), output.shape(), "statement shapes don't match");
        Self {
            output: Box::new(output),
            expression: Box::new(expression),
        }
    }
}

// NOTE: Node implementation.

impl<T, B> Node<T, B> for Statement<T, B>
where
    B: Backend + 'static,
    T: StorageType,
{
    /// Returns the number of elements in the output tensor.
    ///
    /// # Returns
    /// The number of elements in the output tensor.
    fn count(&self) -> usize {
        self.output.count()
    }

    /// Returns the shape of the output tensor as a slice of dimensions.
    ///
    /// # Returns
    /// A slice representing the dimensions of the output tensor.
    fn shape(&self) -> &[usize] {
        self.output.shape()
    }

    /// Returns a boxed clone of the `Statement` instance.
    ///
    /// # Returns
    /// A boxed clone of the `Statement` instance.
    fn clone_box(&self) -> Box<dyn Node<T, B>> {
        Box::new(self.clone())
    }
}

// NOTE: Clone implementation.

impl<T, B> Clone for Statement<T, B>
where
    B: Backend + 'static,
    T: StorageType,
{
    /// Creates a clone of the `Statement` instance.
    ///
    /// # Returns
    /// A clone of the `Statement` instance.
    fn clone(&self) -> Self {
        Self {
            expression: self.expression.clone_box(),
            output: self.output.clone_box(),
        }
    }
}
