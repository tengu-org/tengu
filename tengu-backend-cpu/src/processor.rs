//! This module defines the `Processor` struct which implements the `Processor` trait from the `tengu_backend` crate.
//! It is responsible for managing sources availabe sources and performing computations for each
//! operation supported by backend tensors.
//!
//! The `Processor` struct makes use of the final tagless style, which is an approach to
//! embedding domain-specific languages (DSLs). This style allows for greater flexibility and extensibility
//! by abstracting over the concrete representations of computations. In this particular case, `Processor`
//! handles AST of tensor operations to collect source tensors and produce declaration and shader body code.

use std::collections::HashSet;

use tengu_backend::Processor as RawProcessor;
use tengu_tensor::{Function, Operator, StorageType, Type, UnaryFn};
use tengu_tensor_cpu::Tensor;

use crate::source::{Equality, Source};
use crate::Backend as CPUBackend;

/// The `Processor` struct is used to manage and process tensor sources and perform tensor computations.
pub struct Processor<'a> {
    visited: HashSet<&'a str>,
    sources: Vec<Source<'a>>,
}

impl<'a> Processor<'a> {
    /// Creates a new `Processor` instance.
    ///
    /// # Returns
    /// A new instance of `Processor`.
    pub fn new(_readouts: &'a HashSet<String>) -> Self {
        Self {
            visited: HashSet::new(),
            sources: Vec::new(),
        }
    }

    /// Returns an iterator over the source tensors acquird from the tensor AST.
    ///
    /// # Returns
    /// An iterator over source tensor references.
    pub fn sources(&'a self) -> impl Iterator<Item = &'a Source<'a>> {
        self.sources.iter()
    }
}

// NOTE: Processor trait implementation

impl<'a> RawProcessor<'a, CPUBackend> for Processor<'a> {
    type Repr = Source<'a>;

    /// Processses the tensor. This is the bottom-level call, so the tensor will be added to the
    /// list of available sources.
    ///
    /// # Parameters
    /// - `tensor`: A reference to the tensor to be stored.
    ///
    /// # Returns
    /// Processor representation of the tensor.
    fn var<T: StorageType>(&mut self, tensor: &'a Tensor<T>) -> Self::Repr {
        use tengu_tensor::Tensor;
        let label = tensor.label().expect("input tensors should have a label");
        let source: Source = tensor.into();
        if !self.visited.contains(label) {
            self.sources.push(source.clone());
            self.visited.insert(label);
        }
        source
    }

    /// Generates the representation for a scalar value.
    ///
    /// # Parameters
    /// - `value`: The scalar value to be represented.
    ///
    /// # Returns
    /// Processor representation of the scalar value as a `[1]`-shaped tensor.
    fn scalar<T: StorageType>(&mut self, value: T) -> Self::Repr {
        Tensor::<T>::repeat("", [1], value).into()
    }

    /// Generates the representation for a unary function applied to an inner expression.
    ///
    /// # Parameters
    /// - `inner`: The inner expression representation.
    /// - `function`: The unary function to apply.
    ///
    /// # Returns
    /// Processor representation of the unary function applied to the inner expression.
    fn unary_fn(&mut self, inner: Self::Repr, function: Function) -> Self::Repr {
        match function {
            Function::Exp => inner.exp(),
            Function::Log => inner.log(),
        }
    }

    /// Generates the representation for a binary operation between two expressions.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side expression representation.
    /// - `rhs`: The right-hand side expression representation.
    /// - `operator`: The binary operator to apply.
    ///
    /// # Returns
    /// Processor representation of the binary operation between the two expressions.
    fn binary(&mut self, lhs: Self::Repr, rhs: Self::Repr, operator: Operator) -> Self::Repr {
        match operator {
            Operator::Add => &lhs + &rhs,
            Operator::Sub => &lhs - &rhs,
            Operator::Mul => &lhs * &rhs,
            Operator::Div => &lhs / &rhs,
            Operator::Eq => lhs.eq(&rhs),
            Operator::Neq => lhs.neq(&rhs),
        }
    }

    /// Generates the representation for the type cast of the inner expression to a specified type.
    ///
    /// # Parameters
    /// - `inner`: The inner expression representation.
    /// - `ty`: The target type to cast to.
    ///
    /// # Returns
    /// Processor representation of the inner expression cast to the specified type.
    fn cast(&mut self, inner: Self::Repr, ty: Type) -> Self::Repr {
        match ty {
            Type::U32 => inner.cast::<u32>(),
            Type::I32 => inner.cast::<i32>(),
            Type::F32 => inner.cast::<f32>(),
            Type::Bool => inner.cast::<bool>(),
        }
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
    fn statement(&mut self, out: Self::Repr, expr: Self::Repr) -> Self::Repr {
        out.copy_from(&expr);
        out
    }

    /// Generates a representation for a block of expressions. For CPU implementation, this is a
    /// no-op.
    ///
    /// # Parameters
    /// - `exprs`: An iterator over expression representations to be included in the block.
    fn block(&mut self, _exprs: impl Iterator<Item = Self::Repr>) {}
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use std::collections::HashSet;
    use std::rc::Rc;

    use tengu_backend::{Backend, Processor};
    use tengu_tensor::{Function, Operator, Tensor, Type};

    use crate::Backend as CPUBackend;

    #[test]
    fn scalar() {
        let probes = HashSet::new();
        let mut processor = CPUBackend.processor(&probes);
        let scalar = processor.scalar(2.37);
        let scalar = scalar.into_owned::<f32>();
        assert_eq!(scalar.shape(), [1]);
        let data = scalar.into_data();
        assert_eq!(data, [2.37]);
    }

    #[test]
    fn cast() {
        let probes = HashSet::new();
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.processor(&probes);
        let a = backend.tensor("a", [2, 2], &[1, 2, 3, 4]);
        let a = processor.var(&a);
        let cast_a = processor.cast(a, Type::F32);
        let cast_a = cast_a.into_owned::<f32>();
        assert_eq!(cast_a.shape(), [2, 2]);
        let data = cast_a.into_data();
        assert_eq!(data, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn unary_fn() {
        let probes = HashSet::new();
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.processor(&probes);
        let a = backend.tensor("a", [1], &[1.0]);
        let a = processor.var(&a);
        let exp = processor.unary_fn(a, Function::Exp);
        let exp = exp.into_owned::<f32>();
        assert_eq!(exp.shape(), [1]);
        let data = exp.into_data();
        assert_eq!(data, [std::f32::consts::E]);
    }

    #[test]
    fn binary() {
        let probes = HashSet::new();
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.processor(&probes);
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let b = backend.tensor("b", [4], &[5, 6, 7, 8]);
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.binary(a, b, Operator::Mul);
        let a_add_b = a_add_b.into_owned::<i32>();
        assert_eq!(a_add_b.shape(), [4]);
        let data = a_add_b.into_data();
        assert_eq!(data, [5, 12, 21, 32]);
    }

    #[test]
    fn statement() {
        let probes = HashSet::new();
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.processor(&probes);
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let b = backend.tensor("b", [4], &[5, 6, 7, 8]);
        let c = backend.zero::<i32>("c", [4]);
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.binary(a, b, Operator::Add);
        let c = processor.var(&c);
        let statement = processor.statement(c, a_add_b);
        let statement = statement.into_owned::<i32>();
        assert_eq!(statement.shape(), [4]);
        let data = statement.into_data();
        assert_eq!(data, [6, 8, 10, 12]);
    }
}
