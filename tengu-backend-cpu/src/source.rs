use crate::tensor::Tensor;

mod binary;
mod cast;
mod copy;
mod unary_fn;

// NOTE: AsSource trait for specialization.

/// Marker tag to represent underlying types for tensor elements that are supported by this
/// backend. It is needed becase there's no negative trait bounds in stable Rust.
pub struct Supported;

/// Marker tag to represent underlying types for tensor elements that are not supported by this
/// backend.
pub struct Unsupported;

pub trait AsSource<T = Supported> {
    fn as_source(&self) -> Source<'_>;
    fn into_source(self) -> Source<'static>;
}

// NOTE: Source implementation.

#[derive(Clone)]
pub enum Source<'a> {
    Borrowed(Borrowed<'a>),
    Owned(Owned),
}

impl<'a> Source<'a> {
    pub fn into_owned(self) -> Owned {
        match self {
            Self::Owned(owned) => owned,
            Self::Borrowed(borrowed) => match borrowed {
                Borrowed::Bool(bool) => Owned::Bool(bool.clone()),
                Borrowed::U32(u32) => Owned::U32(u32.clone()),
                Borrowed::I32(i32) => Owned::I32(i32.clone()),
                Borrowed::F32(f32) => Owned::F32(f32.clone()),
            },
        }
    }

    pub fn variant(&self) -> &'static str {
        match self {
            Self::Owned(owned) => owned.variant(),
            Self::Borrowed(borrowed) => borrowed.variant(),
        }
    }
}

// NOTE: Borrowed and Owned implementation.

#[derive(Clone)]
pub enum Borrowed<'a> {
    Bool(&'a Tensor<bool>),
    U32(&'a Tensor<u32>),
    I32(&'a Tensor<i32>),
    F32(&'a Tensor<f32>),
}

impl<'a> Borrowed<'a> {
    pub fn variant(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
        }
    }
}

#[derive(Clone)]
pub enum Owned {
    Bool(Tensor<bool>),
    U32(Tensor<u32>),
    I32(Tensor<i32>),
    F32(Tensor<f32>),
}

impl Owned {
    pub fn lift(self) -> Source<'static> {
        Source::Owned(self)
    }

    pub fn variant(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
        }
    }
}

// NOTE: From implementation.

macro_rules! impl_from {
    ( $type:ty, $variant:ident ) => {
        impl<'a> From<&'a Tensor<$type>> for Source<'a> {
            fn from(value: &'a Tensor<$type>) -> Self {
                Source::Borrowed(Borrowed::$variant(value))
            }
        }
        impl<'a> From<Tensor<$type>> for Source<'a> {
            fn from(value: Tensor<$type>) -> Self {
                Source::Owned(Owned::$variant(value))
            }
        }
    };
}

impl_from!(bool, Bool);
impl_from!(u32, U32);
impl_from!(i32, I32);
impl_from!(f32, F32);
