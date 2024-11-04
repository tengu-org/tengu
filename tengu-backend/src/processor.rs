//! This module defines the `Processor` trait, which is used for processing the abstract syntax tree (AST)
//! of tensor expressions in a final tagless style. Implementations of the `Processor` trait are responsible
//! for transforming or evaluating the AST nodes according to specific backend requirements.

use std::{
    borrow::Cow,
    ops::{Add, Div, Mul, Sub},
};

use tengu_tensor::*;

use crate::Backend;

pub type Atom<'a, B, T> = Option<Cow<'a, <B as Backend>::Tensor<T>>>;

/// The `Processor` trait defines a set of operations for processing tensor expressions.
/// Types that implement this trait can produce and accept representations of tensor expressions
/// and perform various operations on them. Each method of implementing processor will accept
/// representations returned by subexpression methods and returns a representation for that particular
/// epxression. This means that recursion is handled on the frontend and the job of the processor
/// is only to produce final representation (hence the final tagless style).
pub trait Processor<'a, B: Backend> {
    /// The type of a statement produces by the `statement` method of the processor.
    type Statement;

    /// The type of a block produced by the `block` method of the processor.
    type Block;

    /// Processes a tensor variable and produces its representation.
    ///
    /// # Parameters
    /// - `tensor`: The tensor variable to be processed.
    ///
    /// # Returns
    /// A representation of the tensor variable.
    fn var<T: StorageType>(&mut self, tensor: &'a B::Tensor<T>) -> Atom<'a, B, T>;

    /// Processes a scalar value and produces its representation.
    ///
    /// # Parameters
    /// - `value`: The scalar value to be processed.
    ///
    /// # Returns
    /// A representation of the scalar value.
    fn scalar<T: StorageType>(&mut self, value: T) -> Atom<'a, B, T>;

    /// Applies a unary function to a representation and produces a new representation.
    ///
    /// # Parameters
    /// - `inner`: The inner representation to which the unary function is applied.
    /// - `function`: The unary function to apply.
    ///
    /// # Returns
    /// A new representation after applying the unary function.
    fn unary_fn<T>(&mut self, inner: Atom<'a, B, T>, function: Function) -> Atom<'a, B, T>
    where
        T: StorageType,
        B::Tensor<T>: UnaryFn,
    {
        let inner = inner?;
        let tensor = match function {
            Function::Exp => inner.exp(),
            Function::Log => inner.log(),
        };
        Some(Cow::Owned(tensor))
    }

    /// Applies a binary operation to two representations and produces a new representation.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side representation.
    /// - `rhs`: The right-hand side representation.
    /// - `operator`: The binary operator to apply.
    ///
    /// # Returns
    /// A new representation after applying the binary operation.
    fn arithmetic<T>(&mut self, lhs: Atom<'a, B, T>, rhs: Atom<'a, B, T>, operator: Operator) -> Atom<'a, B, T>
    where
        T: StorageType,
        T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
        B::Tensor<T>: Arithmetic,
    {
        let lhs = lhs?;
        let rhs = rhs?;
        let tensor = match operator {
            Operator::Add => lhs.add(&rhs),
            Operator::Sub => lhs.sub(&rhs),
            Operator::Mul => lhs.mul(&rhs),
            Operator::Div => lhs.div(&rhs),
        };
        Some(Cow::Owned(tensor))
    }
    fn relation<T>(&mut self, lhs: Atom<'a, B, T>, rhs: Atom<'a, B, T>, relation: Relation) -> Atom<'a, B, bool>
    where
        T: StorageType + PartialEq,
        B::Tensor<T>: Relational<Output = B::Tensor<bool>>,
    {
        let lhs = lhs?;
        let rhs = rhs?;
        let tensor = match relation {
            Relation::Eq => lhs.eq(&rhs),
            Relation::Neq => lhs.neq(&rhs),
        };
        Some(Cow::Owned(tensor))
    }

    /// Creates a representation of a type cast applyied to a tensor expression.
    ///
    /// # Parameters
    /// - `inner`: The inner representation to be cast.
    /// - `ty`: The target type to cast to.
    ///
    /// # Returns
    /// A new representation after casting.
    fn cast<S, T>(&mut self, inner: Atom<'a, B, T>) -> Atom<'a, B, S>
    where
        T: StorageType,
        S: StorageType,
        B::Tensor<T>: Cast<S, Output = B::Tensor<S>>,
    {
        let tensor = (*inner?).cast();
        Some(Cow::Owned(tensor))
    }

    /// Creates a representation of a statement that assigns an expression to an output.
    ///
    /// # Parameters
    /// - `out`: The output representation.
    /// - `expr`: The expression representation.
    ///
    /// # Returns
    /// A representation of the statement.
    fn statement<T: StorageType>(&mut self, out: &'a B::Tensor<T>, expr: Atom<'a, B, T>) -> Self::Statement
    where
        B::Tensor<T>: CopyFrom;

    /// Processes a block of expressions.
    ///
    /// # Parameters
    /// - `exprs`: An iterator over the expressions to be processed.
    fn block(&mut self, exprs: impl IntoIterator<Item = Self::Statement>) -> Self::Block;

    /// Processes a link.
    ///
    /// # Parameters
    /// - `from`: A reference to the source tensor.
    /// - `to`: A reference to the destination tensor.
    fn link<T: StorageType>(&mut self, from: &'a B::Tensor<T>, to: &'a B::Tensor<T>);
}
