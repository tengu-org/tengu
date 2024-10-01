use std::fmt::Display;

use crate::tensor::Tensor;
use crate::visitor::Visitor;

use binary::Binary;
use cast::Cast;
use traits::{Emit, Node, Shape, Source};

mod binary;
mod cast;
mod ops;
pub mod traits;

// Expression

pub enum Expression<T> {
    Scalar(T),
    Tensor(Tensor<T>),
    Binary(Binary<T>),
    Cast(Cast<T>),
}

impl<T: Clone + Display + 'static> Expression<T> {
    pub fn label(&self) -> Option<&str> {
        match self {
            Self::Tensor(tensor) => Some(tensor.label()),
            _ => None,
        }
    }

    pub fn cast<S>(self) -> Expression<S> {
        Expression::Cast(Cast::new(self))
    }

    pub(crate) fn unify<'a>(&'a self, other: &'a Self) -> Option<&'a [usize]> {
        match (self, other) {
            (Self::Scalar(_), _) => Some(other.shape()),
            (_, Self::Scalar(_)) => Some(self.shape()),
            _ => (self.shape() == other.shape()).then(|| self.shape()),
        }
    }
}

// Node implementation

impl<T> Shape for Expression<T> {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Scalar(_) => &[1],
            Self::Tensor(tensor) => tensor.shape(),
            Self::Binary(binary) => binary.shape(),
            Self::Cast(cast) => cast.shape(),
        }
    }

    fn count(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Tensor(tensor) => tensor.count(),
            Self::Binary(binary) => binary.count(),
            Self::Cast(cast) => cast.count(),
        }
    }
}

impl<T> Emit for Expression<T> {
    fn emit(&self) -> String {
        String::new()
    }
}

impl<T: Clone + Display + 'static> Node for Expression<T> {
    fn visit<'a>(&'a self, visitor: &mut Visitor<'a>) {
        match self {
            Self::Scalar(_) => {}
            Self::Tensor(tensor) => visitor.add(tensor),
            Self::Binary(binary) => binary.visit(visitor),
            Self::Cast(cast) => cast.visit(visitor),
        }
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

// Clone

impl<T: Clone + 'static> Clone for Expression<T> {
    fn clone(&self) -> Self {
        match self {
            Self::Scalar(scalar) => Self::Scalar(scalar.clone()),
            Self::Tensor(tensor) => Self::Tensor(tensor.clone()),
            Self::Binary(binary) => Self::Binary(binary.clone()),
            Self::Cast(cast) => Self::Cast(cast.clone()),
        }
    }
}
