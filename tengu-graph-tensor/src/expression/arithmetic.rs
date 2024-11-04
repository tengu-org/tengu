//! This module defines the `Binary` struct and associated functionality for handling binary operations
//! such as addition, subtraction, multiplication, division, and equality comparisons in tensor expressions.
//! It leverages the backend processing capabilities to apply these operations on tensor data.

use tengu_backend::Backend;
use tengu_tensor::{Operator, StorageType, Unify};

use crate::node::Node;

use super::Expression;

// NOTE: Binary expression implementation.

/// Struct representing a binary operation applied to two tensor expressions.
pub struct Arithmetic<T, B> {
    operator: Operator,
    shape: Vec<usize>,
    count: usize,
    lhs: Box<dyn Node<T, B>>,
    rhs: Box<dyn Node<T, B>>,
}

impl<T, B> Arithmetic<T, B>
where
    B: Backend + 'static,
    T: StorageType,
{
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
    fn new<S1: StorageType, S2: StorageType>(
        operator: Operator,
        lhs: Expression<T, S1, B>,
        rhs: Expression<T, S2, B>,
    ) -> Self {
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

// NOTE: Node implementation.

impl<T, B> Node<T, B> for Arithmetic<T, B>
where
    B: Backend + 'static,
    T: StorageType,
{
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

    /// Returns a boxed clone of the `Binary` instance.
    ///
    /// # Returns
    /// A boxed clone of the `Binary` instance.
    fn clone_box(&self) -> Box<dyn Node<T, B>> {
        Box::new(self.clone())
    }
}

// NOTE: Clone implementation.

impl<T, B> Clone for Arithmetic<T, B>
where
    B: Backend + 'static,
    T: StorageType,
{
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

// NOTE: Arithmetic expression constructors.

impl<T, B> Arithmetic<T, B>
where
    B: Backend + 'static,
    T: StorageType,
{
    /// Creates a new `Binary` instance for addition.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Expression` instance with the addition operation.
    pub fn add<S1: StorageType, S2: StorageType>(
        lhs: Expression<T, S1, B>,
        rhs: Expression<T, S2, B>,
    ) -> Expression<T, T, B> {
        Expression::Arithmetic(Self::new(Operator::Add, lhs, rhs))
    }

    /// Creates a new `Binary` instance for subtraction.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Expression` instance with the subtraction operation.
    pub fn sub<S1: StorageType, S2: StorageType>(
        lhs: Expression<T, S1, B>,
        rhs: Expression<T, S2, B>,
    ) -> Expression<T, T, B> {
        Expression::Arithmetic(Self::new(Operator::Sub, lhs, rhs))
    }

    /// Creates a new `Binary` instance for multiplication.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Expression` instance with the multiplication operation.
    pub fn mul<S1: StorageType, S2: StorageType>(
        lhs: Expression<T, S1, B>,
        rhs: Expression<T, S2, B>,
    ) -> Expression<T, T, B> {
        Expression::Arithmetic(Self::new(Operator::Mul, lhs, rhs))
    }

    /// Creates a new `Binary` instance for division.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Expression` instance with the division operation.
    pub fn div<S1: StorageType, S2: StorageType>(
        lhs: Expression<T, S1, B>,
        rhs: Expression<T, S2, B>,
    ) -> Expression<T, T, B> {
        Expression::Arithmetic(Self::new(Operator::Div, lhs, rhs))
    }
}
