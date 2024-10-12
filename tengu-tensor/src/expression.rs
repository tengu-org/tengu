use as_any::AsAny;
use tengu_backend::{Backend, Processor, StorageType};

use crate::tensor::Tensor;
use crate::Result;
use cast::Cast;
use ops::Binary;
use statement::Statement;
use unary_fn::UnaryFn;

mod binary;
mod cast;
mod ops;
mod statement;
mod unary_fn;

// Node traits

pub trait Source<B: Backend>: AsAny {
    fn matches_to(&self, other: &dyn Source<B>) -> Result<bool>;
    fn copy(&self, to: &dyn Source<B>, linker: &mut B::Linker<'_>) -> Result<()>;
}

pub trait Shape {
    fn shape(&self) -> &[usize];
    fn count(&self) -> usize;
}

pub trait Node<B: Backend>: Shape {
    fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor>::Repr;
    fn find<'a>(&'a self, label: &str) -> Option<&'a dyn Source<B>>;
    fn clone_box(&self) -> Box<dyn Node<B>>;
}

// Expression

pub enum Expression<T: StorageType, B: Backend + 'static>
where
    T: StorageType,
    B: Backend + 'static,
{
    Scalar(T),
    Tensor(Tensor<T, B>),
    Binary(Binary<B>),
    Cast(Cast<T, B>),
    UnaryFn(UnaryFn<B>),
    Statement(Statement<B>),
}

impl<T, B> Expression<T, B>
where
    T: StorageType,
    B: Backend + 'static,
{
    pub fn label(&self) -> Option<&str> {
        match self {
            Self::Tensor(tensor) => Some(tensor.label()),
            _ => None,
        }
    }

    pub fn cast<S: StorageType>(self) -> Expression<S, B> {
        Expression::Cast(Cast::new(self))
    }

    pub fn log(self) -> Expression<T, B> {
        Self::UnaryFn(UnaryFn::log(self))
    }

    pub fn exp(self) -> Expression<T, B> {
        Self::UnaryFn(UnaryFn::exp(self))
    }

    pub fn statement(expr: Expression<T, B>, output: Expression<T, B>) -> Self {
        Self::Statement(Statement::new(expr, output))
    }
}

// Shape implementation

impl<T, B> Shape for Expression<T, B>
where
    T: StorageType,
    B: Backend + 'static,
{
    fn shape(&self) -> &[usize] {
        match self {
            Self::Scalar(_) => &[1],
            Self::Tensor(tensor) => tensor.shape(),
            Self::Binary(binary) => binary.shape(),
            Self::Cast(cast) => cast.shape(),
            Self::UnaryFn(unary_fn) => unary_fn.shape(),
            Self::Statement(statement) => statement.shape(),
        }
    }

    fn count(&self) -> usize {
        match self {
            Self::Scalar(_) => 1,
            Self::Tensor(tensor) => tensor.count(),
            Self::Binary(binary) => binary.count(),
            Self::Cast(cast) => cast.count(),
            Self::UnaryFn(unary_fn) => unary_fn.count(),
            Self::Statement(statement) => statement.count(),
        }
    }
}

// Node implementation

impl<T, B> Node<B> for Expression<T, B>
where
    T: StorageType,
    B: Backend + 'static,
{
    fn clone_box(&self) -> Box<dyn Node<B>> {
        Box::new(self.clone())
    }

    fn find<'a>(&'a self, label: &str) -> Option<&'a dyn Source<B>> {
        match self {
            Self::Scalar(_) => None,
            Self::Tensor(tensor) => (tensor.label() == label).then_some(tensor),
            Self::Binary(binary) => binary.find(label),
            Self::Cast(cast) => cast.find(label),
            Self::UnaryFn(unary_fn) => unary_fn.find(label),
            Self::Statement(statement) => statement.find(label),
        }
    }

    fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor>::Repr {
        match self {
            Self::Scalar(scalar) => processor.scalar(*scalar),
            Self::Tensor(tensor) => processor.var(tensor.raw_tensor()),
            Self::Binary(binary) => binary.visit(processor),
            Self::Cast(cast) => cast.visit(processor),
            Self::UnaryFn(unary_fn) => unary_fn.visit(processor),
            Self::Statement(statement) => statement.visit(processor),
        }
    }
}

// Clone

impl<T: StorageType, B: Backend> Clone for Expression<T, B> {
    fn clone(&self) -> Self {
        match self {
            Self::Scalar(scalar) => Self::Scalar(*scalar),
            Self::Tensor(tensor) => Self::Tensor(tensor.clone()),
            Self::Binary(binary) => Self::Binary(binary.clone()),
            Self::Cast(cast) => Self::Cast(cast.clone()),
            Self::UnaryFn(unary_fn) => Self::UnaryFn(unary_fn.clone()),
            Self::Statement(statement) => Self::Statement(statement.clone()),
        }
    }
}
