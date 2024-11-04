//! Module for defining and manipulating expressions in the Tengu tensor computation framework.
//!
//! This module defines the `Expression` enum and associated traits for creating and managing
//! various types of expressions, including scalar values, tensors, binary operations, casts,
//! unary functions, and statements. It provides a comprehensive interface for constructing
//! and processing these expressions.

use tengu_backend::Backend;
use tengu_tensor::StorageType;

use crate::node::Node;
use crate::Tensor;

use arithmetic::Arithmetic;
use cast::Cast;
use relational::Relational;
use statement::Statement;
use unary_fn::UnaryFn;

mod arithmetic;
mod cast;
mod ops;
mod relational;
mod statement;
mod unary_fn;

// NOTE: Expression implementation.

/// An enum representing various types of expressions in the Tengu framework.
///
/// The `Expression` enum can represent scalar values, tensors, binary operations, casts,
/// unary functions, and statements.
pub enum Expression<T, S, B>
where
    T: StorageType,
    S: StorageType,
    B: Backend + 'static,
{
    Scalar(T),
    Tensor(Tensor<T, B>),
    Arithmetic(Arithmetic<T, B>),
    Relational(Relational<S, B>),
    Cast(Cast<T, S, B>),
    UnaryFn(UnaryFn<T, B>),
    Statement(Statement<T, B>),
}

impl<T, S, B> Expression<T, S, B>
where
    T: StorageType,
    S: StorageType,
    B: Backend + 'static,
{
    /// Returns the label of the expression if it is a tensor.
    ///
    /// # Returns
    /// An optional reference to the label string if the epxression is a tensor, None otherwise.
    pub fn label(&self) -> Option<&str> {
        match self {
            Self::Tensor(tensor) => tensor.label(),
            _ => None,
        }
    }

    /// Create the cast expression.
    ///
    /// # Parameters
    /// - `S`: The target storage type.
    ///
    /// # Returns
    /// A new cast expression with the target storage type.
    pub fn cast<U: StorageType>(self) -> Expression<U, T, B> {
        Expression::Cast(Cast::new(self))
    }

    /// Create the log expression.
    ///
    /// # Returns
    /// A new expression representing the logarithm of the original expression.
    pub fn log(self) -> Expression<T, S, B> {
        Self::UnaryFn(UnaryFn::log(self))
    }

    /// Create the exponential expression.
    ///
    /// # Returns
    /// A new expression representing the exponential of the original expression.
    pub fn exp(self) -> Expression<T, S, B> {
        Self::UnaryFn(UnaryFn::exp(self))
    }

    /// Creates a statement expression.
    ///
    /// # Parameters
    /// - `expr`: The input expression.
    /// - `output`: The output expression.
    ///
    /// # Returns
    /// A new statement expression.
    pub fn statement(expr: Expression<T, S, B>, output: Expression<T, S, B>) -> Self {
        Self::Statement(Statement::new(expr, output))
    }
}

// NOTE: Node implementation.

impl<T, S, B> Node<T, B> for Expression<T, S, B>
where
    T: StorageType,
    S: StorageType,
    B: Backend + 'static,
{
    /// Returns the shape of the expression.
    ///
    /// # Returns
    /// A slice representing the shape of the expression. Scalars have shape of `[1]`.
    fn shape(&self) -> &[usize] {
        match self {
            Self::Scalar(_) => &[1],
            Self::Tensor(tensor) => tensor.shape(),
            Self::Arithmetic(arithmetic) => arithmetic.shape(),
            Self::Relational(relational) => relational.shape(),
            Self::Cast(cast) => cast.shape(),
            Self::UnaryFn(unary_fn) => unary_fn.shape(),
            Self::Statement(statement) => statement.shape(),
        }
    }
    /// Returns the number of elements in the expression.
    ///
    /// # Returns
    /// The number of elements.
    fn count(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Tensor(tensor) => tensor.count(),
            Self::Arithmetic(arithmetic) => arithmetic.count(),
            Self::Relational(relational) => relational.count(),
            Self::Cast(cast) => cast.count(),
            Self::UnaryFn(unary_fn) => unary_fn.count(),
            Self::Statement(statement) => statement.count(),
        }
    }

    /// Clones the expression into a boxed trait object.
    ///
    /// # Returns
    /// A boxed trait object containing the cloned expression.
    fn clone_box(&self) -> Box<dyn Node<T, B>> {
        Box::new(self.clone())
    }
}

// NOTE: Clone implementation.

impl<T, S, B> Clone for Expression<T, S, B>
where
    T: StorageType,
    S: StorageType,
    B: Backend,
{
    /// Clones the expression.
    ///
    /// # Returns
    /// A new expression that is a clone of the original.
    fn clone(&self) -> Self {
        match self {
            Self::Scalar(scalar) => Self::Scalar(*scalar),
            Self::Tensor(tensor) => Self::Tensor(tensor.clone()),
            Self::Arithmetic(arithmetic) => Self::Arithmetic(arithmetic.clone()),
            Self::Relational(relational) => Self::Relational(relational.clone()),
            Self::Cast(cast) => Self::Cast(cast.clone()),
            Self::UnaryFn(unary_fn) => Self::UnaryFn(unary_fn.clone()),
            Self::Statement(statement) => Self::Statement(statement.clone()),
        }
    }
}
