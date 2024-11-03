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

use super::Atom;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashSet};
use std::rc::Rc;
use tengu_tensor_wgpu::Tensor;

use tengu_backend::Processor as RawProcessor;
use tengu_tensor::StorageType;
use tengu_wgpu::Device;

use crate::source::Source;
use crate::Backend as WGPUBackend;

mod block;
mod statement;

pub use block::Block;
use statement::Statement;

const GROUP: usize = 0;

/// The `Processor` struct is used to manage and process tensor sources, bind them, and generate shader code.
/// It holds an emitter for generating code, a declarator for managing variable declarations, and keeps track of various states.
pub struct Processor<'a> {
    device: Rc<Device>,
    visited: HashSet<&'a str>,
    sources: BTreeMap<usize, &'a dyn Source>,
    declarations: Vec<String>,
    current_binding: usize,
}

impl<'a> Processor<'a> {
    /// Creates a new `Processor` instance.
    ///
    /// # Parameters
    /// - `readouts`: A reference to a set of readout labels.
    ///
    /// # Returns
    /// A new instance of `Processor`.
    pub fn new(device: &Rc<Device>) -> Self {
        Self {
            device: Rc::clone(device),
            visited: HashSet::new(),
            sources: BTreeMap::new(),
            declarations: Vec::new(),
            current_binding: 0,
        }
    }

    /// Returns an iterator over the source tensors acquird from the tensor AST.
    ///
    /// # Returns
    /// An iterator over source tensor references.
    pub fn sources(&'a self) -> impl Iterator<Item = &'a dyn Source> {
        self.sources.values().copied()
    }
}

// NOTE: Processor trait implementation

impl<'a> RawProcessor<'a, WGPUBackend> for Processor<'a> {
    type Statement = Statement;
    type Block = Block;

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
    fn var<T: StorageType>(&mut self, tensor: &'a Tensor<T>) -> Atom<'a, T> {
        use tengu_tensor::Tensor;
        let label = Tensor::label(tensor).expect("input tensors should have a label");
        if !self.visited.contains(label) {
            self.sources.insert(self.current_binding, tensor);
            self.declarations.push(tensor.declaration(GROUP, self.current_binding));
            self.visited.insert(label);
            self.current_binding += 1;
        }
        Some(Cow::Borrowed(tensor))
    }

    fn scalar<T: StorageType>(&mut self, value: T) -> Atom<'a, T> {
        let tensor = Tensor::scalar(&self.device, value);
        Some(Cow::Owned(tensor))
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
    fn statement<T: StorageType>(&mut self, out: &'a Tensor<T>, expr: Atom<'a, T>) -> Self::Statement {
        let expr = expr.expect("statements expects some expression");
        Statement::new(out, expr)
    }

    /// Generates a representation for a block of expressions. This is the top-level call and it
    /// will set internal structures such as shader code to their final value.
    ///
    /// # Parameters
    /// - `exprs`: An iterator over expression representations to be included in the block.
    fn block(&mut self, exprs: impl IntoIterator<Item = Self::Statement>) -> Block {
        Block::new(self.declarations.drain(..), exprs)
    }

    fn link<T: StorageType>(&mut self, _from: &'a Tensor<T>, _to: &'a Tensor<T>) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operation::compute::Compute as WGPUCompute;
    use crate::Backend as WGPUBackend;

    use indoc::indoc;
    use regex::RegexSet;
    use tengu_backend::{Backend, Compute};
    use tengu_tensor::{Function, Operator, Relation};

    #[tokio::test]
    async fn scalar() {
        let backend = WGPUBackend::new().await.unwrap();
        let operation = backend.operation::<WGPUCompute>("compute");
        let mut processor = operation.processor();
        let scalar = processor.scalar(2.37);
        let scalar = scalar.unwrap().into_owned();
        assert_eq!(scalar.emit(), "2.37");
    }

    #[tokio::test]
    async fn cast() {
        let backend = WGPUBackend::new().await.unwrap();
        let operation = backend.operation::<WGPUCompute>("compute");
        let mut processor = operation.processor();
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let a = processor.var(&a);
        let cast_a = processor.cast::<f32, _>(a);
        let cast_a = cast_a.unwrap().into_owned();
        assert_eq!(cast_a.emit(), "f32(a[idx])");
    }

    #[tokio::test]
    async fn unary_fn() {
        let backend = WGPUBackend::new().await.unwrap();
        let operation = backend.operation::<WGPUCompute>("compute");
        let mut processor = operation.processor();
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let a = processor.var(&a);
        let cast_a = processor.unary_fn(a, Function::Exp);
        let cast_a = cast_a.unwrap().into_owned();
        assert_eq!(cast_a.emit(), "exp(a[idx])");
    }

    #[tokio::test]
    async fn arithmetic() {
        let backend = WGPUBackend::new().await.unwrap();
        let operation = backend.operation::<WGPUCompute>("compute");
        let mut processor = operation.processor();
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let b = backend.tensor("b", [4], &[5, 6, 7, 8]);
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.arithmetic(a, b, Operator::Mul);
        let a_add_b = a_add_b.unwrap().into_owned();
        assert_eq!(a_add_b.emit(), "(a[idx] * b[idx])");
    }

    #[tokio::test]
    async fn relational() {
        let backend = WGPUBackend::new().await.unwrap();
        let operation = backend.operation::<WGPUCompute>("compute");
        let mut processor = operation.processor();
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let b = backend.tensor("b", [4], &[5, 6, 7, 8]);
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_eq_b = processor.relation(a, b, Relation::Eq);
        let a_eq_b = a_eq_b.unwrap().into_owned();
        assert_eq!(a_eq_b.emit(), "(a[idx] == b[idx])");
    }

    #[tokio::test]
    async fn statement() {
        let backend = WGPUBackend::new().await.unwrap();
        let operation = backend.operation::<WGPUCompute>("compute");
        let mut processor = operation.processor();
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let b = backend.tensor("b", [4], &[5, 6, 7, 8]);
        let c = backend.zero::<i32>("c", [4]);
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.arithmetic(a, b, Operator::Add);
        let c = processor.var(&c);
        let c = c.unwrap().into_owned();
        let statement = processor.statement(&c, a_add_b);
        assert_eq!(statement.expression(), "c[idx] = (a[idx] + b[idx]);");
    }

    #[tokio::test]
    async fn body() {
        let backend = WGPUBackend::new().await.unwrap();
        let operation = backend.operation::<WGPUCompute>("compute");
        let mut processor = operation.processor();
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let b = backend.tensor("b", [4], &[5, 6, 7, 8]);
        let c = backend.zero::<i32>("c", [4]);
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.arithmetic(a, b, Operator::Add);
        let c = processor.var(&c);
        let c = c.unwrap().into_owned();
        let statement = processor.statement(&c, a_add_b);
        let block = processor.block(std::iter::once(statement));
        assert_eq!(
            block.body(),
            indoc!(
                r"
                @compute
                @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let idx = global_id.x;
                    c[idx] = (a[idx] + b[idx]);
                }"
            )
        );
    }

    #[tokio::test]
    async fn declaration() {
        let backend = WGPUBackend::new().await.unwrap();
        let operation = backend.operation::<WGPUCompute>("compute");
        let mut processor = operation.processor();
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let b = backend.tensor("b", [4], &[5, 6, 7, 8]);
        let c = backend.zero::<i32>("c", [4]);
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.arithmetic(a, b, Operator::Add);
        let c = processor.var(&c);
        let c = c.unwrap().into_owned();
        let statement = processor.statement(&c, a_add_b);
        let block = processor.block(std::iter::once(statement));
        let header = block.header();
        let declarations = header.lines().collect::<Vec<_>>();
        let re = RegexSet::new([
            r"@group\(0\) @binding\(\d+\) var<storage, read> a: array<f32>;",
            r"@group\(0\) @binding\(\d+\) var<storage, read> b: array<f32>;",
            r"@group\(0\) @binding\(\d+\) var<storage, read_write> c: array<f32>;",
        ])
        .unwrap();
        for declaration in declarations {
            println!("{:?}", declaration);
            assert!(re.is_match(declaration));
        }
    }
}
