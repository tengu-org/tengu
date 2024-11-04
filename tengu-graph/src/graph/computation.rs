//! Module for defining and managing computations in the Tengu tensor computation framework.
//!
//! This module defines the `Computation` struct and associated methods for creating, managing, and processing
//! computational expressions within the Tengu framework. A computation is a statement executed
//! within a block. It has direct access to all other tensors used inside a block and create a new
//! tensor as a result.

use tengu_backend::{Backend, Processor};
use tengu_graph_tensor::{Expression, StorageType};

use crate::collector::Collector;
use crate::source::Source;

/// A struct representing a computation in the Tengu framework.
///
/// The `Computation` struct holds a computational statement and provides methods to visit and find sources within the statement.
pub struct Computation<B> {
    statement: Box<dyn Node<B>>,
}

impl<B: Backend + 'static> Computation<B> {
    /// Creates a new `Computation` instance with the specified output and expression.
    ///
    /// # Type Parameters
    /// - `T`: The storage type of the expression.
    ///
    /// # Parameters
    /// - `out`: The output expression.
    /// - `expr`: The input expression.
    ///
    /// # Returns
    /// A new `Computation` instance.
    pub fn new<T: StorageType>(out: Expression<T, B>, expr: Expression<T, B>) -> Self {
        let statement = Box::new(Expression::statement(out, expr));
        Self { statement }
    }

    /// Visits the computation with a processor.
    ///
    /// # Parameters
    /// - `processor`: A mutable reference to the processor.
    ///
    /// # Returns
    /// The inner representation used by the processor.
    pub fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor<B>>::Repr {
        self.statement.visit(processor)
    }

    /// Collect the computation with a collector.
    ///
    /// # Parameters
    /// - `collector`: A mutable reference to the collector.
    pub fn collect<'a>(&'a self, collector: &mut Collector<'a, B>) {
        self.statement.collect(collector);
    }

    /// Finds a source within the computation by its label.
    ///
    /// # Parameters
    /// - `label`: The label of the source to find.
    ///
    /// # Returns
    /// An optional reference to the source.
    pub(crate) fn source(&self, label: &str) -> Option<&dyn Source<B>> {
        self.statement.find(label)
    }
}

// NOTE: Shape trait implementation.

impl<B> Shape for Computation<B> {
    /// Returns the shape of the computation.
    ///
    /// # Returns
    /// A slice representing the shape of the computation.
    fn shape(&self) -> &[usize] {
        self.statement.shape()
    }

    /// Returns the number of elements in the computation.
    ///
    /// # Returns
    /// The count of elements.
    fn count(&self) -> usize {
        self.statement.count()
    }
}

// Tests

#[cfg(test)]
mod tests {
    use crate::Tengu;

    use super::*;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn computation_shape_and_count() {
        let tengu = Tengu::wgpu().await.unwrap();
        let a = tengu.tensor([2, 2]).init(&[1.0, 2.0, 3.0, 4.0]);
        let b = tengu.tensor([2, 2]).init(&[5.0, 6.0, 7.0, 8.0]);
        let c = tengu.tensor([2, 2]).zero();
        let computation = Computation::new(c, a + b);
        assert_eq!(computation.shape(), [2, 2]);
        assert_eq!(computation.count(), 4);
    }
}
