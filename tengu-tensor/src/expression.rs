pub use add::AddExpression;
pub use sub::SubExpression;

use crate::{Tensor, WGSLType};

mod add;
mod sub;

#[derive(Clone)]
pub enum Expression<T> {
    Scalar(T),
    Tensor(Tensor<T>),
    Add(AddExpression<T>),
    Sub(SubExpression<T>),
}

impl<T: WGSLType> Expression<T> {
    pub fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut inputs = Vec::new();
        self.collect_inputs(&mut inputs);
        inputs
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Scalar(_) => &[1],
            Self::Tensor(tensor) => tensor.shape(),
            Self::Add(add) => add.shape(),
            Self::Sub(sub) => sub.shape(),
        }
    }

    pub fn count(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Tensor(tensor) => tensor.count(),
            Self::Add(add) => add.count(),
            Self::Sub(sub) => sub.count(),
        }
    }

    pub fn label(&self) -> &str {
        match self {
            Self::Tensor(tensor) => tensor.label(),
            _ => "",
        }
    }

    pub fn emit(&self) -> String {
        match self {
            Self::Scalar(scalar) => scalar.to_string(),
            Self::Tensor(tensor) => tensor.emit(),
            Self::Add(add) => add.emit(),
            Self::Sub(sub) => sub.emit(),
        }
    }

    fn collect_inputs<'a>(&'a self, inputs: &mut Vec<&'a Tensor<T>>) {
        match self {
            Self::Scalar(_) => {}
            Self::Tensor(tensor) => inputs.push(tensor),
            Self::Add(add) => add.collect_inputs(inputs),
            Self::Sub(sub) => sub.collect_inputs(inputs),
        }
    }
}
