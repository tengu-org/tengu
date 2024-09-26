use add::AddExpression;

use crate::{Computation, Tensor};

pub mod add;

pub enum Expression<T> {
    Tensor(Tensor<T>),
    Add(AddExpression<T>),
}

impl<T> Expression<T> {
    pub fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut inputs = Vec::new();
        self.collect_inputs(&mut inputs);
        inputs
    }

    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Tensor(tensor) => tensor.shape(),
            Self::Add(add) => add.shape(),
        }
    }

    pub fn count(&self) -> usize {
        match self {
            Self::Tensor(tensor) => tensor.count(),
            Self::Add(add) => add.count(),
        }
    }

    fn collect_inputs<'a>(&'a self, inputs: &mut Vec<&'a Tensor<T>>) {
        match self {
            Self::Tensor(tensor) => inputs.push(tensor),
            Self::Add(add) => add.collect_inputs(inputs),
        }
    }
}

impl<T> Computation for Expression<T> {
    fn emit(&self, idx: &str) -> String {
        match self {
            Self::Tensor(tensor) => tensor.emit(idx),
            Self::Add(add) => add.emit(idx),
        }
    }
}
