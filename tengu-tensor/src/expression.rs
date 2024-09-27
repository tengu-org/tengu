use add::AddExpression;

use crate::Tensor;

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

    pub fn emit(&self) -> String {
        match self {
            Self::Tensor(tensor) => tensor.emit(),
            Self::Add(add) => add.emit(),
        }
    }

    fn collect_inputs<'a>(&'a self, inputs: &mut Vec<&'a Tensor<T>>) {
        match self {
            Self::Tensor(tensor) => inputs.push(tensor),
            Self::Add(add) => add.collect_inputs(inputs),
        }
    }
}

// Constructors

impl<T> Expression<T> {
    pub fn tensor(tensor: Tensor<T>) -> Self {
        Self::Tensor(tensor)
    }

    pub fn add(lhs: Tensor<T>, rhs: Tensor<T>) -> Self {
        let lhs = Self::tensor(lhs);
        let rhs = Self::tensor(rhs);
        let add_expression = AddExpression::new(lhs, rhs);
        Self::Add(add_expression)
    }
}
