//! This module defines the `Statement` struct and associated functionality for handling
//! tensor statements within expressions. It ensures that tensor shapes match and provides
//! methods for processing and visiting tensor expressions.

use tengu_backend::{Backend, Processor, StorageType};

use super::{Expression, Node, Shape};

/// Struct representing a tensor statement, containing an output tensor and an expression tensor.
pub struct Statement<B: Backend> {
    output: Box<dyn Node<B>>,
    expression: Box<dyn Node<B>>,
}

impl<B: Backend + 'static> Statement<B> {
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
    pub fn new<T: StorageType>(output: Expression<T, B>, expression: Expression<T, B>) -> Self {
        assert_eq!(expression.shape(), output.shape(), "statement shapes don't match");
        Self {
            output: Box::new(output),
            expression: Box::new(expression),
        }
    }
}

// NOTE: Shape implementation.

impl<B: Backend> Shape for Statement<B> {
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
}

// NOTE: Node implementation.

impl<B: Backend + 'static> Node<B> for Statement<B> {
    /// Returns a boxed clone of the `Statement` instance.
    ///
    /// # Returns
    /// A boxed clone of the `Statement` instance.
    fn clone_box(&self) -> Box<dyn Node<B>> {
        Box::new(self.clone())
    }

    /// Finds a source node by its label in the output or expression tensors.
    ///
    /// # Parameters
    /// - `label`: The label of the source node to find.
    ///
    /// # Returns
    /// An optional reference to the found source node.
    fn find<'a>(&'a self, label: &str) -> Option<&'a dyn super::Source<B>> {
        self.output.find(label).or_else(|| self.expression.find(label))
    }

    /// Visits the node with the given processor and processes the tensor statement.
    ///
    /// # Parameters
    /// - `processor`: The processor used to visit the node.
    ///
    /// # Returns
    /// The inner representation used by the processor.
    fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor>::Repr {
        let output = self.output.visit(processor);
        let expression = self.expression.visit(processor);
        processor.statement(output, expression)
    }
}

impl<B: Backend> Clone for Statement<B> {
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
