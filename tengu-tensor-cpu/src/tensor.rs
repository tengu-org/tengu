//! This module provides the implementation of the `Tensor` struct, which represents a tensor on the CPU backend.
//! It includes functionality for creating tensors, and for copying and extracting their data.

use std::borrow::Cow;
use std::cell::RefCell;

use tengu_tensor::StorageType;
use tengu_tensor::Tensor as RawTensor;
use tengu_utils::Label;

mod arithmetic;
mod cast;
mod copy_from;
mod relational;
mod unary_fn;

/// Represents a tensor on the CPU backend.
pub struct Tensor<T> {
    label: Option<Label>,
    count: usize,
    shape: Vec<usize>,
    data: RefCell<Vec<T>>,
}

impl<T: StorageType> Tensor<T> {
    /// Creates a new `Tensor` with the specified backend, label, shape, and data.
    ///
    /// # Parameters
    /// - `backend`: A reference-counted pointer to the WGPU backend.
    /// - `label`: A string label for identifying the tensor.
    /// - `shape`: The shape of the tensor as a vector of unsigned integers.
    /// - `data`: The data to be stored in the tensor.
    ///
    /// # Returns
    /// A new instance of `Tensor`.
    pub fn new(label: impl Into<Label>, shape: impl Into<Vec<usize>>, data: impl Into<Vec<T>>) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            label: Some(label.into()),
            count,
            shape,
            data: data.into().into(),
        }
    }

    pub fn from_tensor<S: StorageType>(other: &Tensor<S>, data: impl Into<Vec<T>>) -> Self {
        Self {
            label: None,
            count: other.count,
            shape: other.shape.clone(),
            data: data.into().into(),
        }
    }

    pub fn from_shape(shape: impl Into<Vec<usize>>, data: impl Into<Vec<T>>) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        let data = data.into();
        assert!(count == data.len(), "data length doesn't match the shape");
        Self {
            label: None,
            count,
            shape,
            data: data.into(),
        }
    }

    /// Creates a new `Tensor` by repeating a single element across a specified shape.
    ///
    /// # Parameters
    /// - `label`: A string label for identifying the tensor.
    /// - `shape`: The shape of the tensor as a vector of unsigned integers.
    /// - `elem`: The element to be repeated.
    ///
    /// # Returns
    /// A new instance of `Tensor` holding `elem` in every cell.
    pub fn repeat(label: impl Into<Label>, shape: impl Into<Vec<usize>>, elem: T) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            label: Some(label.into()),
            count,
            shape,
            data: vec![elem; count].into(),
        }
    }

    /// Consumes this tensor and returns the inner data buffer.
    ///
    /// # Returns
    /// The inner data buffer of the tensor.
    pub fn into_data(self) -> Vec<T> {
        self.data.into_inner()
    }
}

// NOTE: Backend tensor trait implementation.

impl<T: StorageType> RawTensor<T> for Tensor<T> {
    /// Returns the label of the tensor.
    ///
    /// # Returns
    /// The label of the tensor.
    fn label(&self) -> Option<&str> {
        self.label.as_ref().map(|label| label.value())
    }

    /// Returns the number of elements in the tensor.
    ///
    /// # Returns
    /// The number of elements in the tensor.
    fn count(&self) -> usize {
        self.count
    }

    /// Returns the shape of the tensor.
    ///
    /// # Returns
    /// The shape of the tensor as a slice of unsigned integers.
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Retrieves the data from the tensor.
    ///
    /// # Returns
    /// A `Cow` containing either a reference or owned buffer with the tensor data.
    async fn retrieve(&self) -> anyhow::Result<Cow<'_, [T::IOType]>> {
        let data = self
            .data
            .borrow()
            .iter()
            .map(|v| v.convert())
            .collect::<Vec<_>>()
            .into();
        Ok(data)
    }
}

// NOTE: Clone implementation.

impl<T: StorageType> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            label: self.label.clone(),
            count: self.count,
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }
}
