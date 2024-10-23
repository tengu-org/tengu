//! This module defines the `Processor` trait, which is used for processing the abstract syntax tree (AST)
//! of tensor expressions in a final tagless style. Implementations of the `Processor` trait are responsible
//! for transforming or evaluating the AST nodes according to specific backend requirements.

use tengu_backend_tensor::{Function, Operator, StorageType, Type};

use crate::Backend;

/// The `Processor` trait defines a set of operations for processing tensor expressions.
/// Types that implement this trait can produce and accept representations of tensor expressions
/// and perform various operations on them. Each method of implementing processor will accept
/// representations returned by subexpression methods and returns a representation for that particular
/// epxression. This means that recursion is handled on the frontend and the job of the processor
/// is only to produce final representation (hence the final tagless style).
pub trait Processor<'a> {
    /// The type of the backend that this processor interacts with.
    type Backend: Backend;

    /// The type of the representation produced and accepted by this processor.
    type Repr;

    /// Processes a tensor variable and produces its representation.
    ///
    /// # Parameters
    /// - `tensor`: The tensor variable to be processed.
    ///
    /// # Returns
    /// A representation of the tensor variable.
    fn var<T: StorageType>(&mut self, tensor: &'a <Self::Backend as Backend>::Tensor<T>) -> Self::Repr;

    /// Processes a scalar value and produces its representation.
    ///
    /// # Parameters
    /// - `value`: The scalar value to be processed.
    ///
    /// # Returns
    /// A representation of the scalar value.
    fn scalar<T: StorageType>(&mut self, value: T) -> Self::Repr;

    /// Applies a unary function to a representation and produces a new representation.
    ///
    /// # Parameters
    /// - `inner`: The inner representation to which the unary function is applied.
    /// - `symbol`: The symbol representing the unary function.
    ///
    /// # Returns
    /// A new representation after applying the unary function.
    fn unary_fn(&mut self, inner: Self::Repr, function: Function) -> Self::Repr;

    /// Applies a binary operation to two representations and produces a new representation.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side representation.
    /// - `rhs`: The right-hand side representation.
    /// - `symbol`: The symbol representing the binary operation.
    ///
    /// # Returns
    /// A new representation after applying the binary operation.
    fn binary(&mut self, lhs: Self::Repr, rhs: Self::Repr, operator: Operator) -> Self::Repr;

    /// Creates a representation of a type cast applyied to a tensor expression.
    ///
    /// # Parameters
    /// - `inner`: The inner representation to be cast.
    /// - `ty`: The target type for the cast.
    ///
    /// # Returns
    /// A new representation after casting.
    fn cast(&mut self, inner: Self::Repr, ty: Type) -> Self::Repr;

    /// Creates a representation of a statement that assigns an expression to an output.
    ///
    /// # Parameters
    /// - `out`: The output representation.
    /// - `expr`: The expression representation.
    ///
    /// # Returns
    /// A representation of the statement.
    fn statement(&mut self, out: Self::Repr, expr: Self::Repr) -> Self::Repr;

    /// Processes a block of expressions.
    ///
    /// # Parameters
    /// - `exprs`: An iterator over the expressions to be processed.
    fn block(&mut self, exprs: impl Iterator<Item = Self::Repr>);
}
