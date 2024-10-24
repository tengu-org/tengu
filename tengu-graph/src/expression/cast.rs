//! This module defines the `Cast` struct and associated functionality for handling type casting in tensor expressions.
//! It leverages the backend processing capabilities to apply type casting on tensor data.

use std::marker::PhantomData;

use tengu_backend::{Backend, Processor};
use tengu_backend_tensor::StorageType;

use super::Expression;
use crate::collector::Collector;
use crate::node::Node;
use crate::shape::Shape;
use crate::source::Source;

/// Struct representing a type cast on a tensor expression.
pub struct Cast<T, B> {
    expression: Box<dyn Node<B>>,
    phantom: PhantomData<T>,
}

impl<T: StorageType, B: Backend + 'static> Cast<T, B> {
    /// Creates a new `Cast` instance.
    ///
    /// # Parameters
    /// - `expr`: The tensor expression to be cast to a different type.
    ///
    /// # Returns
    /// A new `Cast` instance.
    pub fn new<S: StorageType>(expr: Expression<S, B>) -> Self {
        Self {
            expression: Box::new(expr),
            phantom: PhantomData,
        }
    }
}

// NOTE: Shape implementation.

impl<T, B> Shape for Cast<T, B> {
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

// NOTE: Node implenentation.

impl<T: StorageType, B> Node<B> for Cast<T, B>
where
    T: Clone + 'static,
    B: Backend + 'static,
{
    /// Returns a boxed clone of the `Cast` instance.
    ///
    /// # Returns
    /// A boxed clone of the `Cast` instance.
    fn clone_box(&self) -> Box<dyn Node<B>> {
        Box::new(self.clone())
    }

    /// Collect sources from the cast operation.
    ///
    /// # Parameters
    /// - `collector`: A mutable reference to the collector.
    fn collect<'a>(&'a self, collector: &mut Collector<'a, B>) {
        self.expression.collect(collector);
    }

    /// Finds a source node by its label in the expression tensor.
    ///
    /// # Parameters
    /// - `label`: The label of the source node to find.
    ///
    /// # Returns
    /// An optional reference to the found source node.
    fn find<'a>(&'a self, label: &str) -> Option<&'a dyn Source<B>> {
        self.expression.find(label)
    }

    /// Visits the node with the given processor and applies the type cast.
    ///
    /// # Parameters
    /// - `processor`: The processor used to visit the node.
    ///
    /// # Returns
    /// The inner representation used by the processor.
    fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor<B>>::Repr {
        let epxression = self.expression.visit(processor);
        processor.cast(epxression, T::as_type())
    }
}

// NOTE: Clone implementation.

impl<T, B: Backend> Clone for Cast<T, B> {
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
