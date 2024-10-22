use std::borrow::Cow;

use crate::tensor::Tensor;

/// Marker tag to represent underlying types for tensor elements that are supported by this
/// backend. It is needed becase there's no negative trait bounds in stable Rust.
pub struct Supported;

/// Marker tag to represent underlying types for tensor elements that are not supported by this
/// backend.
pub struct Unsupported;

type TensorCow<'a, T> = Cow<'a, Tensor<T>>;

#[derive(Clone)]
pub enum Source<'a> {
    Bool(TensorCow<'a, bool>),
    U32(TensorCow<'a, u32>),
    I32(TensorCow<'a, i32>),
    F32(TensorCow<'a, f32>),
}

pub trait AsSource<T = Supported> {
    fn as_source(&self) -> Source<'_>;
}
