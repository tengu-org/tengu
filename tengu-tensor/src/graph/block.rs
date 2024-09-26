use std::sync::Arc;

use crate::{expression::Expression, Computation, Tengu, Tensor};

pub struct Block<T> {
    expression: Expression<T>,
    output: Tensor<T>,
}

impl<T> Block<T> {
    pub fn new(tengu: Arc<Tengu>, expression: Expression<T>) -> Self {
        let output = tengu.tensor(expression.shape()).empty();
        Self { expression, output }
    }

    fn declaration(&self) -> String {
        let group = 0; // TODO: Should we increment this somewhere?
        self.expression
            .inputs()
            .iter()
            .enumerate()
            .map(|(binding, tensor)| tensor.declaration(group, binding))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn body(&self, idx: &str) -> String {
        format!(
            r"
            @compute
            @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                {} = {}
            }}",
            self.output.emit(idx),
            self.expression.emit(idx)
        )
    }
}

impl<T> Computation for Block<T> {
    fn emit(&self, idx: &str) -> String {
        let declaration = self.declaration();
        let body = self.body(idx);
        format!("{declaration}\n{body}")
    }
}
