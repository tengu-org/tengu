//! This module defines the `Processor` struct which implements the `Processor` trait from the `tengu_backend` crate.
//! It is responsible for managing sources, binding them, and generating shader code using a declarator and emitter.
//!
//! The `Processor` struct makes use of the final tagless style, which is an approach to
//! embedding domain-specific languages (DSLs). This style allows for greater flexibility and extensibility
//! by abstracting over the concrete representations of computations. In this particular case, `Processor`
//! handles AST of tensor operations to collect source tensors and produce declaration and shader body code.
//!
//! The main responsibilities of the `Processor` include:
//! - Managing and binding tensor sources.
//! - Generating shader code using the `Emitter` and `Declarator` components.
//! - Providing various operations such as variable binding, scalar representation, unary and binary functions,
//!   type casting, and block generation.

use std::collections::{BTreeMap, HashSet};
use tengu_backend::{Backend, StorageType};

use crate::{source::Source, Backend as WGPUBackend};
use declarator::Declarator;
use emitter::Emitter;

mod declarator;
mod emitter;

/// The `Processor` struct is used to manage and process tensor sources, bind them, and generate shader code.
/// It holds an emitter for generating code, a declarator for managing variable declarations, and keeps track of various states.
pub struct Processor<'a> {
    emitter: Emitter,
    declarator: Declarator<'a>,
    element_count: usize,
    shader: String,
    bound_sources: HashSet<&'a str>,
    sources: BTreeMap<usize, &'a dyn Source>,
    current_binding: usize,
}

impl<'a> Processor<'a> {
    /// Creates a new `Processor` instance.
    ///
    /// # Returns
    /// A new instance of `Processor`.
    pub fn new() -> Self {
        Self {
            emitter: Emitter::new(),
            declarator: Declarator::new(),
            element_count: 0,
            shader: String::new(),
            bound_sources: HashSet::new(),
            sources: BTreeMap::new(),
            current_binding: 0,
        }
    }

    /// Returns the maximum number of elements use by the tensors in the AST.
    ///
    /// # Returns
    /// The number of elements as a `usize`.
    pub fn element_count(&self) -> usize {
        self.element_count
    }

    /// Returns an iterator over the source tensors acquird from the tensor AST.
    ///
    /// # Returns
    /// An iterator over source tensor references.
    pub fn sources(&'a self) -> impl Iterator<Item = &'a dyn Source> {
        self.sources.values().copied()
    }

    /// Returns the generated shader code as a string slice.
    ///
    /// # Returns
    /// A string slice with the final shader code.
    pub fn shader(&self) -> &str {
        &self.shader
    }
}

// Processor trait implementation

impl<'a> tengu_backend::Processor<'a> for Processor<'a> {
    type Backend = WGPUBackend;
    type Repr = (usize, String);

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
        if !self.bound_sources.contains(tensor.label()) {
            self.declarator.var(self.current_binding, tensor);
            self.sources.insert(self.current_binding, tensor);
            self.bound_sources.insert(tensor.label());
            self.current_binding += 1;
        }
        (tensor.count(), self.emitter.var(tensor))
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
        self.element_count = 0;
        (0, self.emitter.scalar(value))
    }

    /// Generates the representation for a unary function applied to an inner expression.
    ///
    /// # Parameters
    /// - `inner`: The inner expression representation.
    /// - `symbol`: The symbol representing the unary function.
    ///
    /// # Returns
    /// A tuple containing the number of elements and the resulting expression's shader representation.
    fn unary_fn(&mut self, inner: Self::Repr, symbol: &str) -> Self::Repr {
        let expression = self.emitter.unary_fn(inner.1, symbol);
        let element_count = inner.0;
        (element_count, expression)
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
    fn binary(&mut self, lhs: Self::Repr, rhs: Self::Repr, symbol: &str) -> Self::Repr {
        let expression = self.emitter.binary(lhs.1, rhs.1, symbol);
        let element_count = lhs.0.max(rhs.0);
        (element_count, expression)
    }

    /// Generates the representation for the type cast of the inner expression to a specified type.
    ///
    /// # Parameters
    /// - `inner`: The inner expression representation.
    /// - `ty`: The target type as a string.
    ///
    /// # Returns
    /// A tuple containing the number of elements and the resulting cast expression's shader representation.
    fn cast(&mut self, inner: Self::Repr, ty: &str) -> Self::Repr {
        let expression = self.emitter.cast(inner.1, ty);
        let element_count = inner.0;
        (element_count, expression)
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
        let expression = self.emitter.statement(out.1, expr.1);
        let element_count = out.0.max(expr.0);
        (element_count, expression)
    }

    /// Generates a representation for a block of expressions. This is the top-level call and it
    /// will set internal structures such as shader code to their final value.
    ///
    /// # Parameters
    /// - `exprs`: An iterator over expression representations to be included in the block.
    fn block(&mut self, exprs: impl Iterator<Item = Self::Repr>) {
        let (count_exprs, emit_exprs): (Vec<_>, Vec<_>) = exprs.unzip();
        self.emitter.block(emit_exprs.into_iter());
        self.element_count = count_exprs
            .into_iter()
            .max()
            .expect("block should have at least one computation");
        let header = self.declarator.header();
        let body = self.emitter.body();
        self.shader = format!("{}\n\n{}", header, body);
    }
}

// Default implementation

impl<'a> Default for Processor<'a> {
    fn default() -> Self {
        Processor::new()
    }
}
