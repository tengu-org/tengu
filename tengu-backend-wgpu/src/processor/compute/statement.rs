use std::borrow::Cow;

use tengu_tensor::StorageType;
use tengu_tensor_wgpu::Tensor;

pub struct Statement {
    expression: String,
    count: usize,
}

impl Statement {
    pub fn new<'a, T: StorageType>(out: &'a Tensor<T>, expr: Cow<'a, Tensor<T>>) -> Self {
        use tengu_tensor::Tensor;
        assert!(out.shape() == expr.shape(), "statement lhs and rhs don't match");
        let expression = format!("{} = {};", out.emit(), expr.emit());
        let count = out.count();
        Self { expression, count }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn expression(&self) -> &str {
        &self.expression
    }
}
