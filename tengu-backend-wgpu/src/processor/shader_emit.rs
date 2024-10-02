use indoc::formatdoc;
use itertools::Itertools;
use tengu_backend::{Backend, StorageType};
use tengu_wgpu::BufferUsage;

use crate::source::Source;
use crate::Backend as WGPUBackend;

const GROUP: usize = 0;

pub struct ShaderEmitProcessor {
    binding: usize,
    declarations: Vec<String>,
    expression: String,
}

impl ShaderEmitProcessor {
    pub fn new() -> Self {
        Self {
            binding: 0,
            declarations: Vec::new(),
            expression: String::new(),
        }
    }

    pub fn shader(&self) -> String {
        let declaration = self.declaration();
        let body = self.body();
        format!("{declaration}\n\n{body}")
    }

    fn declaration(&self) -> String {
        self.declarations.join("\n")
    }

    fn body(&self) -> String {
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

impl<'a> tengu_backend::Processor<'a> for ShaderEmitProcessor {
    type Backend = WGPUBackend;
    type Repr = String;

    fn var<T: StorageType>(&mut self, tensor: &'a <Self::Backend as Backend>::Tensor<T>) -> Self::Repr {
        let label = tensor.label();
        let access = access(tensor.buffer().usage());
        let ty = std::any::type_name::<T>();
        let binding = self.binding;
        let declaration = format!("@group({GROUP}) @binding({binding}) var<storage, {access}> {label}: array<{ty}>;");
        self.declarations.push(declaration);
        self.binding += 1;
        format!("{}[idx]", tensor.label())
    }

    fn scalar<T: StorageType>(&mut self, value: T) -> Self::Repr {
        value.to_string()
    }

    fn unary_fn(&mut self, inner: Self::Repr, symbol: &str) -> Self::Repr {
        format!("{symbol}({inner})")
    }

    fn binary(&mut self, lhs: Self::Repr, rhs: Self::Repr, symbol: &str) -> Self::Repr {
        format!("({lhs} {symbol} {rhs})")
    }

    fn cast(&mut self, inner: Self::Repr, ty: &str) -> Self::Repr {
        format!("{ty}({inner})")
    }

    fn statement(&mut self, out: Self::Repr, expr: Self::Repr) -> Self::Repr {
        format!("{out} = {expr};")
    }

    fn block(&mut self, exprs: impl Iterator<Item = Self::Repr>) {
        self.expression = exprs.into_iter().join("    \n");
    }
}

fn access(usage: BufferUsage) -> &'static str {
    match usage {
        BufferUsage::Read => "read",
        BufferUsage::Write => "write",
        BufferUsage::ReadWrite => "read_write",
        BufferUsage::Staging => panic!("cannot declare a staging buffer in a shader"),
    }
}

// Default implementation

impl Default for ShaderEmitProcessor {
    fn default() -> Self {
        ShaderEmitProcessor::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend as WGPUBackend;

    use indoc::indoc;
    use regex::RegexSet;
    use tengu_backend::{Backend, Processor};

    #[tokio::test]
    async fn emit() {
        let backend = WGPUBackend::new().await.unwrap();
        let a = backend.tensor("b", &[1, 2, 3, 4]);
        let b = backend.tensor("c", &[5, 6, 7, 8]);
        let c = backend.zero::<u32>("a", 4);
        let mut processor = ShaderEmitProcessor::new();
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.binary(a, b, "+");
        let c = processor.var(&c);
        let statement = processor.statement(c, a_add_b);
        assert_eq!(statement, "c[idx] = (a[idx] + b[idx]);");
    }

    #[tokio::test]
    async fn declaration() {
        let backend = WGPUBackend::new().await.unwrap();
        let a = backend.tensor("a", &[1.0, 2.0, 3.0, 4.0]);
        let b = backend.tensor("b", &[5.0, 6.0, 7.0, 8.0]);
        let c = backend.zero::<u32>("a", 4);
        let mut processor = ShaderEmitProcessor::new();
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.binary(a, b, "+");
        let c = processor.var(&c);
        let _statement = processor.statement(c, a_add_b);
        let declaration = processor.declaration();
        let declarations = declaration.lines().collect::<Vec<_>>();
        let re = RegexSet::new([
            r"@group\(0\) @binding\(\d+\) var<storage, read> a: array<f32>;",
            r"@group\(0\) @binding\(\d+\) var<storage, read> b: array<f32>;",
            r"@group\(0\) @binding\(\d+\) var<storage, read_write> out: array<f32>;",
        ])
        .unwrap();
        for declaration in declarations {
            println!("{:?}", declaration);
            assert!(re.is_match(declaration));
        }
    }

    #[tokio::test]
    async fn body() {
        let backend = WGPUBackend::new().await.unwrap();
        let a = backend.tensor("b", &[1, 2, 3, 4]);
        let b = backend.tensor("c", &[5, 6, 7, 8]);
        let c = backend.zero::<u32>("a", 4);
        let mut processor = ShaderEmitProcessor::new();
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
                    out[idx] = (a[idx] + b[idx]);
                }"
            )
        );
    }
}
