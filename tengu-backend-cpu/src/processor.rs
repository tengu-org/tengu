use std::collections::HashSet;

use tengu_backend::Backend;
use tengu_backend_tensor::{Function, Operator, StorageType, Type, UnaryFn};

use crate::source::{AsSource, Equality, Source};
use crate::tensor::Tensor;
use crate::Backend as CPUBackend;

// NOTE: Processor implementation.

pub struct Processor<'a> {
    visited: HashSet<&'a str>,
    sources: Vec<Source<'a>>,
    readouts: &'a HashSet<String>,
    readout_sources: Vec<Source<'a>>,
}

impl<'a> Processor<'a> {
    /// Creates a new `Processor` instance.
    ///
    /// # Returns
    /// A new instance of `Processor`.
    pub fn new(readouts: &'a HashSet<String>) -> Self {
        Self {
            visited: HashSet::new(),
            sources: Vec::new(),
            readouts,
            readout_sources: Vec::new(),
        }
    }

    /// Returns an iterator over the source tensors acquird from the tensor AST.
    ///
    /// # Returns
    /// An iterator over source tensor references.
    pub fn sources(&'a self) -> impl Iterator<Item = &'a Source<'a>> {
        self.sources.iter()
    }

    /// Returns an iterator over the source tensors acquird from the tensor AST that can be used in
    /// readout operations.
    ///
    /// # Returns
    /// An iterator over source tensor references.
    pub fn readout_sources(&'a self) -> impl Iterator<Item = &'a Source<'a>> {
        self.readout_sources.iter()
    }
}

// NOTE: Processor trait implementation

impl<'a> tengu_backend::Processor<'a> for Processor<'a> {
    type Backend = CPUBackend;
    type Repr = Source<'a>;

    /// Processses the tensor. This is the bottom-level call, so the tensor will be added to the
    /// list of available sources, and it will bee used to generate a declaration and part of the
    /// shader body. Each uniquely labeled tensor will get a different binding.
    ///
    /// # Parameters
    /// - `tensor`: A reference to the tensor to be bound.
    ///
    /// # Returns
    /// Processor representation of the tensor, consisting of the number of elements in the tensor
    /// and emitted shader representation of the tensor.
    fn var<T: StorageType>(&mut self, tensor: &'a <Self::Backend as Backend>::Tensor<T>) -> Self::Repr {
        use tengu_backend_tensor::Tensor;
        let label = tensor.label();
        let source = tensor.as_source();
        if !self.visited.contains(label) {
            self.sources.push(source.clone());
            self.visited.insert(label);
            if self.readouts.contains(label) {
                self.readout_sources.push(source.clone());
            }
        }
        source
    }

    /// Generates the representation for a scalar value.
    ///
    /// # Parameters
    /// - `value`: The scalar value to be represented.
    ///
    /// # Returns
    /// A tuple containing the number of elements (always 0 for scalars) and its shader representation,
    /// which in this case will be a literal.
    fn scalar<T: StorageType>(&mut self, value: T) -> Self::Repr {
        Tensor::<T>::repeat("label", [1], value).into_source()
    }

    /// Generates the representation for a unary function applied to an inner expression.
    ///
    /// # Parameters
    /// - `inner`: The inner expression representation.
    /// - `symbol`: The symbol representing the unary function.
    ///
    /// # Returns
    /// A tuple containing the number of elements and the resulting expression's shader representation.
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
    /// - `symbol`: The symbol representing the binary operation.
    ///
    /// # Returns
    /// A tuple containing the maximum count of elements between the two expressions and the resulting
    /// expression's shader representation.
    fn binary(&mut self, lhs: Self::Repr, rhs: Self::Repr, operator: Operator) -> Self::Repr {
        match operator {
            Operator::Add => &lhs + &rhs,
            Operator::Sub => &lhs - &rhs,
            Operator::Mul => &lhs * &rhs,
            Operator::Div => &lhs / &rhs,
            Operator::Eq => lhs.eq(&rhs),
            Operator::Neq => lhs.eq(&rhs),
        }
    }

    /// Generates the representation for the type cast of the inner expression to a specified type.
    ///
    /// # Parameters
    /// - `inner`: The inner expression representation.
    /// - `ty`: The target type as a string.
    ///
    /// # Returns
    /// A tuple containing the number of elements and the resulting cast expression's shader representation.
    fn cast(&mut self, inner: Self::Repr, ty: Type) -> Self::Repr {
        match ty {
            Type::Bool => inner.cast_bool(),
            Type::U32 => inner.cast_u32(),
            Type::I32 => inner.cast_i32(),
            Type::F32 => inner.cast_f32(),
        }
    }

    /// Generates the representation of a statement combining an output and an expression.
    ///
    /// # Parameters
    /// - `out`: The output expression representation.
    /// - `expr`: The input expression representation.
    ///
    /// # Returns
    /// A tuple containing the maximum count of elements between the two expressions and the resulting
    /// statement's shader representation.
    fn statement(&mut self, out: Self::Repr, expr: Self::Repr) -> Self::Repr {
        out.copy_from(&expr);
        out
    }

    /// Generates a representation for a block of expressions. This is the top-level call and it
    /// will set internal structures such as shader code to their final value.
    ///
    /// # Parameters
    /// - `exprs`: An iterator over expression representations to be included in the block.
    fn block(&mut self, _exprs: impl Iterator<Item = Self::Repr>) {}
}
