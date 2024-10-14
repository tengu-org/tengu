//! This module defines the `Emitter` struct and associated functions for generating
//! shader code in a WGPU-based tensor computation framework.
//!
//! The `Emitter` struct is used to create and manage expressions and statements
//! for use in compute shaders, in the tensor body, not in the variable declaration part.

use indoc::formatdoc;
use itertools::Itertools;
use tengu_backend::StorageType;

use crate::source::Source;
use crate::tensor::Tensor;

pub struct Emitter {
    expression: String,
}

/// A struct for generating shader code expressions and statements.
impl Emitter {
    /// Creates a new `Emitter`.
    ///
    /// # Returns
    /// A new instance of `Emitter`.
    pub fn new() -> Self {
        Self {
            expression: String::new(),
        }
    }

    /// Generates the body of the compute shader.
    ///
    /// # Returns
    /// A `String` containing the shader body with all expressions.
    pub fn body(&self) -> String {
        formatdoc!(
            r"
            @compute
            @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                {}
            }}",
            self.expression,
        )
    }
}

// NOTE: Processing interface

impl Emitter {
    /// Returns a string representation of the tensor variable.
    ///
    /// # Parameters
    /// - `tensor`: The tensor to declare as a variable.
    ///
    /// # Returns
    /// A `String` representing the variable declaration.
    pub fn var<T: StorageType>(&mut self, tensor: &Tensor<T>) -> String {
        format!("{}[idx]", tensor.label())
    }

    /// Return a string representation of a scalar literal.
    ///
    /// # Parameters
    /// - `value`: The scalar value.
    ///
    /// # Returns
    /// A `String` representing the scalar value.
    pub fn scalar<T: StorageType>(&mut self, value: T) -> String {
        value.to_string()
    }

    /// Returns a string representation of unary function expression.
    ///
    /// # Parameters
    /// - `inner`: The inner expression.
    /// - `symbol`: The unary function symbol.
    ///
    /// # Returns
    /// A `String` representing the unary function application.
    pub fn unary_fn(&mut self, inner: String, symbol: &str) -> String {
        format!("{symbol}({inner})")
    }

    /// Returns a string representation of a binary expression.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side expression.
    /// - `rhs`: The right-hand side expression.
    /// - `symbol`: The binary operator symbol.
    ///
    /// # Returns
    /// A `String` representing the binary operation.
    pub fn binary(&mut self, lhs: String, rhs: String, symbol: &str) -> String {
        format!("({lhs} {symbol} {rhs})")
    }

    /// Returns a string representation of a cast expression.
    ///
    /// # Parameters
    /// - `inner`: The inner expression.
    /// - `ty`: The target type.
    ///
    /// # Returns
    /// A `String` representing the casted expression.
    pub fn cast(&mut self, inner: String, ty: &str) -> String {
        format!("{ty}({inner})")
    }

    /// Return a string representation of a statement.
    ///
    /// # Parameters
    /// - `out`: The output variable.
    /// - `expr`: The expression to assign to the output.
    ///
    /// # Returns
    /// A `String` representing the statement.
    pub fn statement(&mut self, out: String, expr: String) -> String {
        format!("{out} = {expr};")
    }

    /// Processes a block of expressions. The final representation is stored inside the emitter.
    ///
    /// # Parameters
    /// - `exprs`: An iterator over expressions to include in the block.
    pub fn block(&mut self, exprs: impl Iterator<Item = String>) {
        self.expression = exprs.into_iter().join("\n    ");
    }
}

// NOTE: Default implementation

impl Default for Emitter {
    /// Creates a new `Emitter` using the default implementation.
    ///
    /// # Returns
    /// A new instance of `Emitter`.
    fn default() -> Self {
        Emitter::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend as WGPUBackend;

    use indoc::indoc;
    use tengu_backend::Backend;

    #[tokio::test]
    async fn scalar() {
        let mut processor = Emitter::new();
        let scalar = processor.scalar(2.37);
        assert_eq!(scalar, "2.37");
    }

    #[tokio::test]
    async fn cast() {
        let backend = WGPUBackend::new().await.unwrap();
        let a = backend.tensor("a", &[1, 2, 3, 4]);
        let mut processor = Emitter::new();
        let a = processor.var(&a);
        let cast_a = processor.cast(a, "f32");
        assert_eq!(cast_a, "f32(a[idx])");
    }

    #[tokio::test]
    async fn unary_fn() {
        let backend = WGPUBackend::new().await.unwrap();
        let a = backend.tensor("a", &[1, 2, 3, 4]);
        let mut processor = Emitter::new();
        let a = processor.var(&a);
        let cast_a = processor.unary_fn(a, "exp");
        assert_eq!(cast_a, "exp(a[idx])");
    }

    #[tokio::test]
    async fn binary() {
        let backend = WGPUBackend::new().await.unwrap();
        let a = backend.tensor("a", &[1, 2, 3, 4]);
        let b = backend.tensor("b", &[5, 6, 7, 8]);
        let mut processor = Emitter::new();
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.binary(a, b, "*");
        assert_eq!(a_add_b, "(a[idx] * b[idx])");
    }

    #[tokio::test]
    async fn statement() {
        let backend = WGPUBackend::new().await.unwrap();
        let a = backend.tensor("a", &[1, 2, 3, 4]);
        let b = backend.tensor("b", &[5, 6, 7, 8]);
        let c = backend.zero::<f32>("c", 4);
        let mut processor = Emitter::new();
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.binary(a, b, "+");
        let c = processor.var(&c);
        let statement = processor.statement(c, a_add_b);
        assert_eq!(statement, "c[idx] = (a[idx] + b[idx]);");
    }

    #[tokio::test]
    async fn body() {
        let backend = WGPUBackend::new().await.unwrap();
        let a = backend.tensor("a", &[1, 2, 3, 4]);
        let b = backend.tensor("b", &[5, 6, 7, 8]);
        let c = backend.zero::<f32>("c", 4);
        let mut processor = Emitter::new();
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.binary(a, b, "+");
        let c = processor.var(&c);
        let statement = processor.statement(c, a_add_b);
        processor.block(std::iter::once(statement));
        assert_eq!(
            processor.body(),
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
}
