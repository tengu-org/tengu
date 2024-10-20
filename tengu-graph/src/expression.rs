//! Module for defining and manipulating expressions in the Tengu tensor computation framework.
//!
//! This module defines the `Expression` enum and associated traits for creating and managing
//! various types of expressions, including scalar values, tensors, binary operations, casts,
//! unary functions, and statements. It provides a comprehensive interface for constructing
//! and processing these expressions.

use tengu_backend::{Backend, Processor};
use tengu_tensor_traits::StorageType;

use cast::Cast;
use ops::Binary;
use statement::Statement;
use unary_fn::UnaryFn;

use crate::collector::Collector;
use crate::node::{Node, Shape};
use crate::source::Source;
use crate::tensor::Tensor;

mod binary;
mod cast;
mod ops;
mod statement;
mod unary_fn;

// NOTE: Expression implementation.

/// An enum representing various types of expressions in the Tengu framework.
///
/// The `Expression` enum can represent scalar values, tensors, binary operations, casts,
/// unary functions, and statements.
pub enum Expression<T: StorageType, B: Backend + 'static>
where
    T: StorageType,
    B: Backend + 'static,
{
    Scalar(T),
    Tensor(Tensor<T, B>),
    Binary(Binary<B>),
    Cast(Cast<T, B>),
    UnaryFn(UnaryFn<B>),
    Statement(Statement<B>),
}

impl<T, B> Expression<T, B>
where
    T: StorageType,
    B: Backend + 'static,
{
    /// Returns the label of the expression if it is a tensor.
    ///
    /// # Returns
    /// An optional reference to the label string if the epxression is a tensor, None otherwise.
    pub fn label(&self) -> Option<&str> {
        match self {
            Self::Tensor(tensor) => Some(tensor.label()),
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
    pub fn cast<S: StorageType>(self) -> Expression<S, B> {
        Expression::Cast(Cast::new(self))
    }

    /// Create the log expression.
    ///
    /// # Returns
    /// A new expression representing the logarithm of the original expression.
    pub fn log(self) -> Expression<T, B> {
        Self::UnaryFn(UnaryFn::log(self))
    }

    /// Create the exponential expression.
    ///
    /// # Returns
    /// A new expression representing the exponential of the original expression.
    pub fn exp(self) -> Expression<T, B> {
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
    pub fn statement(expr: Expression<T, B>, output: Expression<T, B>) -> Self {
        Self::Statement(Statement::new(expr, output))
    }
}

// NOTE: Shape implementation.

impl<T, B> Shape for Expression<T, B>
where
    T: StorageType,
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
            Self::Binary(binary) => binary.shape(),
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
            Self::Binary(binary) => binary.count(),
            Self::Cast(cast) => cast.count(),
            Self::UnaryFn(unary_fn) => unary_fn.count(),
            Self::Statement(statement) => statement.count(),
        }
    }
}

// NOTE: Node implementation.

impl<T, B> Node<B> for Expression<T, B>
where
    T: StorageType,
    B: Backend + 'static,
{
    /// Clones the expression into a boxed trait object.
    ///
    /// # Returns
    /// A boxed trait object containing the cloned expression.
    fn clone_box(&self) -> Box<dyn Node<B>> {
        Box::new(self.clone())
    }

    fn collect<'a>(&'a self, collector: &mut Collector<'a, B>) {
        match self {
            Self::Scalar(_) => {}
            Self::Tensor(tensor) => collector.add(tensor),
            Self::Binary(binary) => binary.collect(collector),
            Self::Cast(cast) => cast.collect(collector),
            Self::UnaryFn(unary_fn) => unary_fn.collect(collector),
            Self::Statement(statement) => statement.collect(collector),
        }
    }

    /// Finds a source within the expression by its label.
    ///
    /// # Parameters
    /// - `label`: The label of the source to find.
    ///
    /// # Returns
    /// An optional reference to the source.
    fn find<'a>(&'a self, label: &str) -> Option<&'a dyn Source<B>> {
        match self {
            Self::Scalar(_) => None,
            Self::Tensor(tensor) => (tensor.label() == label).then_some(tensor),
            Self::Binary(binary) => binary.find(label),
            Self::Cast(cast) => cast.find(label),
            Self::UnaryFn(unary_fn) => unary_fn.find(label),
            Self::Statement(statement) => statement.find(label),
        }
    }

    /// Visits the expression with a processor.
    ///
    /// # Parameters
    /// - `processor`: A mutable reference to the processor.
    ///
    /// # Returns
    /// The inner representation used by the processor.
    fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor>::Repr {
        match self {
            Self::Scalar(scalar) => processor.scalar(*scalar),
            Self::Tensor(tensor) => processor.var(tensor.raw()),
            Self::Binary(binary) => binary.visit(processor),
            Self::Cast(cast) => cast.visit(processor),
            Self::UnaryFn(unary_fn) => unary_fn.visit(processor),
            Self::Statement(statement) => statement.visit(processor),
        }
    }
}

// NOTE: Clone implementation.

impl<T: StorageType, B: Backend> Clone for Expression<T, B> {
    /// Clones the expression.
    ///
    /// # Returns
    /// A new expression that is a clone of the original.
    fn clone(&self) -> Self {
        match self {
            Self::Scalar(scalar) => Self::Scalar(*scalar),
            Self::Tensor(tensor) => Self::Tensor(tensor.clone()),
            Self::Binary(binary) => Self::Binary(binary.clone()),
            Self::Cast(cast) => Self::Cast(cast.clone()),
            Self::UnaryFn(unary_fn) => Self::UnaryFn(unary_fn.clone()),
            Self::Statement(statement) => Self::Statement(statement.clone()),
        }
    }
}
