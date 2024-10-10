use indoc::formatdoc;
use itertools::Itertools;
use tengu_backend::StorageType;

use crate::source::Source;
use crate::tensor::Tensor;

pub struct Emitter {
    expression: String,
}

impl Emitter {
    pub fn new() -> Self {
        Self {
            expression: String::new(),
        }
    }

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

// Processing interface

impl Emitter {
    pub fn var<T: StorageType>(&mut self, tensor: &Tensor<T>) -> String {
        format!("{}[idx]", tensor.label())
    }

    pub fn scalar<T: StorageType>(&mut self, value: T) -> String {
        value.to_string()
    }

    pub fn unary_fn(&mut self, inner: String, symbol: &str) -> String {
        format!("{symbol}({inner})")
    }

    pub fn binary(&mut self, lhs: String, rhs: String, symbol: &str) -> String {
        format!("({lhs} {symbol} {rhs})")
    }

    pub fn cast(&mut self, inner: String, ty: &str) -> String {
        format!("{ty}({inner})")
    }

    pub fn statement(&mut self, out: String, expr: String) -> String {
        format!("{out} = {expr};")
    }

    pub fn block(&mut self, exprs: impl Iterator<Item = String>) {
        self.expression = exprs.into_iter().join("\n    ");
    }
}

// Default implementation

impl Default for Emitter {
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
