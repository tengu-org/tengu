//! This module defines the `Source` trait which is used to represent a data source that allows
//! treating tensors in type-independent fashion. It relies heavly on the `Cage` struct to store
//! tensors as `dyn Any` objects and maintain the type information in enum variants.

use std::any::TypeId;
use tengu_backend_tensor::StorageType;
use tengu_utils::Cage;

use crate::tensor::Tensor;

mod arithmetic;
mod cast;
mod copy;
mod relational;
mod unary_fn;

pub use relational::Equality;

/// The `Source` trait represents a "type-less" tensor. It is used by the `Processor`
/// to handle all tensors in a uniform fashion.
pub enum Source<'a> {
    /// A source variant storing a u32-based tensor.
    U32(Cage<'a>),
    /// A source variant storing a i32-based tensor.
    I32(Cage<'a>),
    /// A source variant storing a f32-based tensor.
    F32(Cage<'a>),
    /// A source variant storing a bool-based tensor.
    Bool(Cage<'a>),
}

impl<'a> Source<'a> {
    /// Consumes the source and returns the owned tensor.
    ///
    /// # Returns
    /// The tensor stored in this source.
    ///
    /// # Panics
    /// If the source type does not match the tensor type the method will panic.
    pub fn into_owned<T: StorageType>(self) -> Tensor<T> {
        let tensor = match self {
            Self::U32(cage) => cage.into_owned::<Tensor<T>>(),
            Self::I32(cage) => cage.into_owned::<Tensor<T>>(),
            Self::F32(cage) => cage.into_owned::<Tensor<T>>(),
            Self::Bool(cage) => cage.into_owned::<Tensor<T>>(),
        };
        tensor.expect("Source type mismatch")
    }

    /// Returns a reference to the tensor stored in this source.
    ///
    /// # Returns
    /// A reference to the tensor stored in this source.
    ///
    /// # Panics
    /// If the source type does not match the tensor type the method will panic.
    pub fn as_ref<T: StorageType>(&self) -> &Tensor<T> {
        let tensor = match self {
            Self::U32(cage) => cage.as_ref::<Tensor<T>>(),
            Self::I32(cage) => cage.as_ref::<Tensor<T>>(),
            Self::F32(cage) => cage.as_ref::<Tensor<T>>(),
            Self::Bool(cage) => cage.as_ref::<Tensor<T>>(),
        };
        tensor.expect("Source type mismatch")
    }

    /// Returns the string representation of the type variant of this `Source`.
    ///
    /// # Returns
    /// A string slice representing the type variant of this `Source`.
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
    /// Converts a tensor into a source. The conversion is done at runtimes asing type information
    /// provided by TypeId facilities. This particular implementation constructs an owned variant
    /// of the `Source`.
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
    /// Converts a tensor into a source. The conversion is done at runtimes asing type information
    /// provided by TypeId facilities. This particular implementation constructs a borrowed variant
    /// of the `Source`.
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
