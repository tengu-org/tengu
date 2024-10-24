use std::any::{Any, TypeId};

use crate::tensor::Tensor;

mod arithmetic;
mod cast;
mod copy;
mod relational;
mod unary_fn;

pub use relational::Equality;
use tengu_backend_tensor::StorageType;

pub enum Cage<'a> {
    Borrowed(&'a dyn Any),
    Owned(Box<dyn Any>),
}

impl<'a> Cage<'a> {
    pub fn owned<T: Clone + 'static>(value: T) -> Self {
        Self::Owned(Box::new(value))
    }

    pub fn borrowed<T: Clone + 'static>(value: &'a T) -> Self {
        Self::Borrowed(value)
    }

    pub fn into_owned<T: Clone + 'static>(self) -> Option<T> {
        match self {
            Self::Borrowed(tensor) => tensor.downcast_ref::<T>().cloned(),
            Self::Owned(tensor) => tensor.downcast::<T>().ok().map(|b| *b),
        }
    }

    pub fn as_ref<T: Clone + 'static>(&self) -> Option<&T> {
        match self {
            Self::Borrowed(tensor) => tensor.downcast_ref::<T>(),
            Self::Owned(tensor) => tensor.downcast_ref::<T>(),
        }
    }

    pub fn cloned<T: Clone + 'static>(&self) -> Option<Self> {
        let inner = self.as_ref::<T>().cloned()?;
        Some(Cage::owned(inner))
    }

    pub fn lift<T: Clone + 'static>(self) -> Self {
        match self {
            Self::Owned(_) => self,
            Self::Borrowed(tensor) => Self::Owned(Box::new(tensor.downcast_ref::<T>().unwrap().clone())),
        }
    }
}

pub enum Source<'a> {
    U32(Cage<'a>),
    I32(Cage<'a>),
    F32(Cage<'a>),
    Bool(Cage<'a>),
}

impl<'a> Source<'a> {
    pub fn into_owned<T: StorageType>(self) -> Tensor<T> {
        let tensor = match self {
            Self::U32(cage) => cage.into_owned::<Tensor<T>>(),
            Self::I32(cage) => cage.into_owned::<Tensor<T>>(),
            Self::F32(cage) => cage.into_owned::<Tensor<T>>(),
            Self::Bool(cage) => cage.into_owned::<Tensor<T>>(),
        };
        tensor.expect("Source type mismatch")
    }

    pub fn as_ref<T: StorageType>(&self) -> &Tensor<T> {
        let tensor = match self {
            Self::U32(cage) => cage.as_ref::<Tensor<T>>(),
            Self::I32(cage) => cage.as_ref::<Tensor<T>>(),
            Self::F32(cage) => cage.as_ref::<Tensor<T>>(),
            Self::Bool(cage) => cage.as_ref::<Tensor<T>>(),
        };
        tensor.expect("Source type mismatch")
    }

    pub fn variant(&self) -> &'static str {
        match self {
            Self::U32(_) => "u32",
            Self::I32(_) => "i32",
            Self::F32(_) => "f32",
            Self::Bool(_) => "bool",
        }
    }
}

impl<'a> Clone for Source<'a> {
    fn clone(&self) -> Self {
        match self {
            Self::U32(cage) => Self::U32(cage.cloned::<Tensor<u32>>().expect("Source type mismatch")),
            Self::I32(cage) => Self::I32(cage.cloned::<Tensor<i32>>().expect("Source type mismatch")),
            Self::F32(cage) => Self::F32(cage.cloned::<Tensor<f32>>().expect("Source type mismatch")),
            Self::Bool(cage) => Self::Bool(cage.cloned::<Tensor<bool>>().expect("Source type mismatch")),
        }
    }
}

impl<'a, T: StorageType> From<Tensor<T>> for Source<'a> {
    fn from(value: Tensor<T>) -> Self {
        if TypeId::of::<T>() == TypeId::of::<u32>() {
            return Source::U32(Cage::owned(value));
        }
        if TypeId::of::<T>() == TypeId::of::<i32>() {
            return Source::I32(Cage::owned(value));
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            return Source::F32(Cage::owned(value));
        }
        if TypeId::of::<T>() == TypeId::of::<bool>() {
            return Source::Bool(Cage::owned(value));
        }
        unreachable!("Source type mismatch")
    }
}

impl<'a, T: StorageType> From<&'a Tensor<T>> for Source<'a> {
    fn from(value: &'a Tensor<T>) -> Self {
        if TypeId::of::<T>() == TypeId::of::<u32>() {
            return Source::U32(Cage::borrowed(value));
        }
        if TypeId::of::<T>() == TypeId::of::<i32>() {
            return Source::I32(Cage::borrowed(value));
        }
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            return Source::F32(Cage::borrowed(value));
        }
        if TypeId::of::<T>() == TypeId::of::<bool>() {
            return Source::Bool(Cage::borrowed(value));
        }
        unreachable!("Source type mismatch")
    }
}
