use add::AddExpression;
use sub::SubExpression;

use crate::Tensor;

pub mod add;
pub mod sub;

pub enum Expression<T> {
    Tensor(Tensor<T>),
    Add(AddExpression<T>),
    Sub(SubExpression<T>),
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
            Self::Sub(sub) => sub.shape(),
        }
    }

    pub fn count(&self) -> usize {
        match self {
            Self::Tensor(tensor) => tensor.count(),
            Self::Add(add) => add.count(),
            Self::Sub(sub) => sub.count(),
        }
    }

    pub fn emit(&self) -> String {
        match self {
            Self::Tensor(tensor) => tensor.emit(),
            Self::Add(add) => add.emit(),
            Self::Sub(sub) => sub.emit(),
        }
    }

    fn collect_inputs<'a>(&'a self, inputs: &mut Vec<&'a Tensor<T>>) {
        match self {
            Self::Tensor(tensor) => inputs.push(tensor),
            Self::Add(add) => add.collect_inputs(inputs),
            Self::Sub(sub) => sub.collect_inputs(inputs),
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

    pub fn sub(lhs: Tensor<T>, rhs: Tensor<T>) -> Self {
        let lhs = Self::tensor(lhs);
        let rhs = Self::tensor(rhs);
        let sub_expression = SubExpression::new(lhs, rhs);
        Self::Sub(sub_expression)
    }
}
