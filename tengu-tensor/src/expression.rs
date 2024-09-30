use binary::Binary;

use crate::{Tensor, WGSLType};

mod binary;
mod ops;

#[derive(Clone)]
pub enum Expression<T> {
    Scalar(T),
    Tensor(Tensor<T>),
    Binary(Binary<T>),
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
            Self::Binary(add) => add.shape(),
        }
    }

    pub fn match_shape(&self, other: &Self) -> bool {
        match self {
            Self::Scalar(_) => true,
            _ => match other {
                Self::Scalar(_) => true,
                _ => self.shape() == other.shape(),
            },
        }
    }

    pub fn count(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Tensor(tensor) => tensor.count(),
            Self::Binary(add) => add.count(),
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
            Self::Binary(add) => add.emit(),
        }
    }

    fn collect_inputs<'a>(&'a self, inputs: &mut Vec<&'a Tensor<T>>) {
        match self {
            Self::Scalar(_) => {}
            Self::Tensor(tensor) => inputs.push(tensor),
            Self::Binary(add) => add.collect_inputs(inputs),
        }
    }
}
