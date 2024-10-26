//! This module defines the `UnaryFn` struct and associated functionality for handling unary functions
//! such as logarithm and exponentiation in tensor expressions. This is a helper struct for storing
//! `UnaryFn` variant on the `Expression` struct.

use tengu_backend::{Backend, Processor};
use tengu_graph_tensor::{Function, StorageType};

use super::Expression;
use crate::collector::Collector;
use crate::node::Node;
use crate::shape::Shape;
use crate::source::Source;

// NOTE: UnaryFn

/// Struct representing a unary function applied to a tensor expression.
pub struct UnaryFn<B> {
    function: Function,
    expression: Box<dyn Node<B>>,
}

impl<B: Backend + 'static> UnaryFn<B> {
    /// Creates a new `UnaryFn` instance.
    ///
    /// # Parameters
    /// - `function`: The unary function to apply.
    /// - `expr`: The tensor expression to which the function is applied.
    ///
    /// # Returns
    /// A new `UnaryFn` instance.
    pub fn new<T: StorageType>(function: Function, expr: Expression<T, B>) -> Self {
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
    pub fn exp<T: StorageType>(expr: Expression<T, B>) -> Self {
        Self::new(Function::Exp, expr)
    }

    /// Creates a new `UnaryFn` instance for the logarithm function.
    ///
    /// # Parameters
    /// - `expr`: The tensor expression to which the logarithm function is applied.
    ///
    /// # Returns
    /// A new `UnaryFn` instance with the logarithm function.
    pub fn log<T: StorageType>(expr: Expression<T, B>) -> Self {
        Self::new(Function::Log, expr)
    }
}

// NOTE: Trait implementations.

impl<B> Shape for UnaryFn<B> {
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
}

impl<B: Backend + 'static> Node<B> for UnaryFn<B> {
    /// Returns a boxed clone of the `UnaryFn` instance.
    ///
    /// # Returns
    /// A boxed clone of the `UnaryFn` instance.
    fn clone_box(&self) -> Box<dyn Node<B>> {
        Box::new(self.clone())
    }

    /// Collect sources from the unary operation.
    ///
    /// # Parameters
    /// - `collector`: A mutable reference to the collector.
    fn collect<'a>(&'a self, collector: &mut Collector<'a, B>) {
        self.expression.collect(collector);
    }

    /// Finds a source node by its label.
    ///
    /// # Parameters
    /// - `label`: The label of the source node to find.
    ///
    /// # Returns
    /// An optional reference to the found source node.
    fn find<'a>(&'a self, label: &str) -> Option<&'a dyn Source<B>> {
        self.expression.find(label)
    }

    /// Visits the node with the given processor and applies the unary function.
    ///
    /// # Parameters
    /// - `processor`: The processor used to visit the node.
    ///
    /// # Returns
    /// The inner representation used by the processor.
    fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor<B>>::Repr {
        let expr = self.expression.visit(processor);
        processor.unary_fn(expr, self.function)
    }
}

// NOTE: Clone implementation.

impl<B: Backend> Clone for UnaryFn<B> {
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
