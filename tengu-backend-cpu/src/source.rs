use crate::tensor::Tensor;

pub struct Supported;
pub struct Unsupported;

// NOTE: SourceType implementation.

#[derive(Clone, Copy)]
pub enum Source<'a> {
    Bool(&'a Tensor<bool>),
    U32(&'a Tensor<u32>),
    I32(&'a Tensor<i32>),
    F32(&'a Tensor<f32>),
}

// NOTE: Source implementation.

pub trait AsSource<T = Supported> {
    fn as_source(&self) -> Source<'_>;
}
