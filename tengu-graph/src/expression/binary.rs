//! This module defines the `Binary` struct and associated functionality for handling binary operations
//! such as addition, subtraction, multiplication, division, and equality comparisons in tensor expressions.
//! It leverages the backend processing capabilities to apply these operations on tensor data.

use tengu_backend::{Backend, Processor};
use tengu_graph_tensor::{Operator, StorageType};

use super::Expression;
use crate::collector::Collector;
use crate::node::Node;
use crate::shape::Shape;
use crate::source::Source;
use crate::unify::Unify;

// NOTE: Binary expression implementation.

/// Struct representing a binary operation applied to two tensor expressions.
pub struct Binary<B> {
    operator: Operator,
    shape: Vec<usize>,
    count: usize,
    lhs: Box<dyn Node<B>>,
    rhs: Box<dyn Node<B>>,
}

impl<B: Backend + 'static> Binary<B> {
    /// Creates a new `Binary` instance.
    ///
    /// # Parameters
    /// - `operator`: The binary operator to apply.
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Binary` instance.
    ///
    /// # Panics
    /// Panics if the shapes of `lhs` and `rhs` do not match.
    fn new<T: StorageType>(operator: Operator, lhs: Expression<T, B>, rhs: Expression<T, B>) -> Self {
        let shape = lhs.shape().unify(rhs.shape()).expect("Shapes don't match");
        let count = shape.iter().product();
        Self {
            operator,
            shape,
            count,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

// NOTE: Shape implementation.

impl<B> Shape for Binary<B> {
    /// Returns the maximum number of elements in lhs and rhs subexpressions.
    ///
    /// # Returns
    /// The number of elements in the tensor.
    fn count(&self) -> usize {
        self.count
    }

    /// Returns the shape of the tensor expression as a slice of dimensions.
    /// This shape is the result of unification on dimensions of lhs and rhs subexpressions.
    ///
    /// # Returns
    /// A slice representing the dimensions of the tensor.
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

// NOTE: Node implementation.

impl<B> Node<B> for Binary<B>
where
    B: Backend + 'static,
{
    /// Returns a boxed clone of the `Binary` instance.
    ///
    /// # Returns
    /// A boxed clone of the `Binary` instance.
    fn clone_box(&self) -> Box<dyn Node<B>> {
        Box::new(self.clone())
    }

    /// Collect sources from the binary operation.
    ///
    /// # Parameters
    /// - `collector`: A mutable reference to the collector.
    fn collect<'a>(&'a self, collector: &mut Collector<'a, B>) {
        self.lhs.collect(collector);
        self.rhs.collect(collector);
    }

    /// Finds a source node by its label in the left-hand or right-hand side tensor expressions.
    ///
    /// # Parameters
    /// - `label`: The label of the source node to find.
    ///
    /// # Returns
    /// An optional reference to the found source node.
    fn find<'a>(&'a self, label: &str) -> Option<&'a dyn Source<B>> {
        self.lhs.find(label).or_else(|| self.rhs.find(label))
    }

    /// Visits the node with the given processor and applies the binary operation.
    ///
    /// # Parameters
    /// - `processor`: The processor used to visit the node.
    ///
    /// # Returns
    /// The inner representation used by the processor.
    fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor<B>>::Repr {
        let lhs = self.lhs.visit(processor);
        let rhs = self.rhs.visit(processor);
        processor.binary(lhs, rhs, self.operator)
    }
}

// NOTE: Clone implementation.

impl<B: Backend> Clone for Binary<B> {
    /// Creates a clone of the `Binary` instance.
    ///
    /// # Returns
    /// A clone of the `Binary` instance.
    fn clone(&self) -> Self {
        Self {
            operator: self.operator,
            shape: self.shape.clone(),
            count: self.count,
            lhs: self.lhs.clone_box(),
            rhs: self.rhs.clone_box(),
        }
    }
}

// NOTE: Binary expression constructors.

impl<B: Backend + 'static> Binary<B> {
    /// Creates a new `Binary` instance for addition.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Expression` instance with the addition operation.
    pub fn add<T: StorageType>(lhs: Expression<T, B>, rhs: Expression<T, B>) -> Expression<T, B> {
        Expression::Binary(Self::new(Operator::Add, lhs, rhs))
    }

    /// Creates a new `Binary` instance for subtraction.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Expression` instance with the subtraction operation.
    pub fn sub<T: StorageType>(lhs: Expression<T, B>, rhs: Expression<T, B>) -> Expression<T, B> {
        Expression::Binary(Self::new(Operator::Sub, lhs, rhs))
    }

    /// Creates a new `Binary` instance for multiplication.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Expression` instance with the multiplication operation.
    pub fn mul<T: StorageType>(lhs: Expression<T, B>, rhs: Expression<T, B>) -> Expression<T, B> {
        Expression::Binary(Self::new(Operator::Mul, lhs, rhs))
    }

    /// Creates a new `Binary` instance for division.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Expression` instance with the division operation.
    pub fn div<T: StorageType>(lhs: Expression<T, B>, rhs: Expression<T, B>) -> Expression<T, B> {
        Expression::Binary(Self::new(Operator::Div, lhs, rhs))
    }

    /// Creates a new `Binary` instance for equality comparison.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Expression` instance with the equality comparison operation.
    pub fn eq<T: StorageType>(lhs: Expression<T, B>, rhs: Expression<T, B>) -> Expression<bool, B> {
        Expression::Binary(Binary::new(Operator::Eq, lhs, rhs))
    }

    /// Creates a new `Binary` instance for inequality comparison.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Expression` instance with the inequality comparison operation.
    pub fn neq<T: StorageType>(lhs: Expression<T, B>, rhs: Expression<T, B>) -> Expression<bool, B> {
        Expression::Binary(Binary::new(Operator::Neq, lhs, rhs))
    }
}

#[cfg(test)]
mod tests {
    use crate::expression::Shape;
    use crate::Tengu;

    #[tokio::test]
    async fn propagation() {
        let tengu = Tengu::wgpu().await.unwrap();
        let lhs = tengu.tensor([4, 1, 3]).zero::<f32>();
        let rhs = tengu.tensor([2, 3]).zero::<f32>();
        let add = lhs + rhs;
        assert_eq!(add.shape(), &[4, 2, 3]);
        assert_eq!(add.count(), 24);
    }
}
