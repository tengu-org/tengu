//! This module defines the `Processor` struct which implements the `Processor` trait from the `tengu_backend` crate.
//! It is responsible for managing sources availabe sources and performing computations for each
//! operation supported by backend tensors.
//!
//! The `Processor` struct makes use of the final tagless style, which is an approach to
//! embedding domain-specific languages (DSLs). This style allows for greater flexibility and extensibility
//! by abstracting over the concrete representations of computations. In this particular case, `Processor`
//! handles AST of tensor operations to collect source tensors and produce declaration and shader body code.

use std::borrow::Cow;
use tengu_backend::Processor as RawProcessor;
use tengu_tensor::*;
use tengu_tensor_cpu::Tensor;

use super::Atom;
use crate::Backend as CPUBackend;

/// The `Processor` struct is used to manage and process tensor sources and perform tensor computations.
pub struct Processor;

// NOTE: Processor trait implementation

impl<'a> RawProcessor<'a, CPUBackend> for Processor {
    type Statement = ();
    type Block = ();

    /// Processses the tensor. This is the bottom-level call, so the tensor will be added to the
    /// list of available sources.
    ///
    /// # Parameters
    /// - `tensor`: A reference to the tensor to be stored.
    ///
    /// # Returns
    /// Processor representation of the tensor.
    fn var<T: StorageType>(&mut self, tensor: &'a Tensor<T>) -> Atom<'a, T> {
        Some(Cow::Borrowed(tensor))
    }

    /// Generates the representation for a scalar value.
    ///
    /// # Parameters
    /// - `value`: The scalar value to be represented.
    ///
    /// # Returns
    /// Processor representation of the scalar value as a `[1]`-shaped tensor.
    fn scalar<T: StorageType>(&mut self, value: T) -> Atom<'a, T> {
        Some(Cow::Owned(Tensor::<T>::repeat("", [1], value)))
    }

    /// Copies the data from the `expr` expression to the `out` resulting source and outputs it as a
    /// method result.
    ///
    /// # Parameters
    /// - `out`: The output expression representation.
    /// - `expr`: The input expression representation.
    ///
    /// # Returns
    /// Representation of the `out` source.
    fn statement<T: StorageType>(&mut self, out: &'a Tensor<T>, expr: Atom<'a, T>) -> Self::Statement {
        if let Some(expr) = expr {
            out.copy_from(&expr, &mut ());
        }
    }

    /// Generates a representation for a block of expressions. For CPU implementation, this is a
    /// no-op.
    ///
    /// # Parameters
    /// - `exprs`: An iterator over expression representations to be included in the block.
    fn block(&mut self, _exprs: impl IntoIterator<Item = Self::Statement>) {}

    fn link<T: StorageType>(&mut self, _from: &'a Tensor<T>, _to: &'a Tensor<T>) {}
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use std::rc::Rc;

    use tengu_backend::{Backend, Compute, Processor};
    use tengu_tensor::{Function, Operator, Relation, Tensor};

    use crate::operation::Compute as CPUCompute;
    use crate::Backend as CPUBackend;

    #[test]
    fn scalar() {
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.operation::<CPUCompute>("compute").processor();
        let scalar = processor.scalar(2.37);
        let scalar = scalar.unwrap().into_owned();
        assert_eq!(scalar.shape(), [1]);
        assert_eq!(scalar.into_data(), [2.37]);
    }

    #[test]
    fn cast() {
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.operation::<CPUCompute>("compute").processor();
        let a = backend.tensor("a", [2, 2], &[1, 2, 3, 4]);
        let a = processor.var(&a);
        let cast_a = processor.cast::<f32, _>(a);
        let cast_a = cast_a.unwrap().into_owned();
        assert_eq!(cast_a.shape(), [2, 2]);
        assert_eq!(cast_a.into_data(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn unary_fn() {
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.operation::<CPUCompute>("compute").processor();
        let a = backend.tensor("a", [1], &[1.0]);
        let a = processor.var(&a);
        let exp = processor.unary_fn(a, Function::Exp);
        let exp = exp.unwrap().into_owned();
        assert_eq!(exp.shape(), [1]);
        assert_eq!(exp.into_data(), [std::f32::consts::E]);
    }

    #[test]
    fn arithmetic() {
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.operation::<CPUCompute>("compute").processor();
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let b = backend.tensor("b", [4], &[5, 6, 7, 8]);
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.arithmetic(a, b, Operator::Mul);
        let a_add_b = a_add_b.unwrap().into_owned();
        assert_eq!(a_add_b.shape(), [4]);
        assert_eq!(a_add_b.into_data(), [5, 12, 21, 32]);
    }

    #[test]
    fn relation() {
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.operation::<CPUCompute>("compute").processor();
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let b = backend.tensor("b", [4], &[1, 2, 3, 4]);
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_eq_b = processor.relation(a, b, Relation::Eq);
        let a_eq_b = a_eq_b.unwrap().into_owned();
        assert_eq!(a_eq_b.shape(), [4]);
        assert_eq!(a_eq_b.into_data(), [true, true, true, true]);
    }

    #[test]
    fn statement() {
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.operation::<CPUCompute>("compute").processor();
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let b = backend.tensor("b", [4], &[5, 6, 7, 8]);
        let c = backend.zero::<i32>("c", [4]);
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.arithmetic(a, b, Operator::Add);
        let c = processor.var(&c);
        let c = c.unwrap().into_owned();
        processor.statement(&c, a_add_b);
        assert_eq!(c.shape(), [4]);
        assert_eq!(c.into_data(), [6, 8, 10, 12]);
    }
}
