//! Module for tensors in the tensor computation framework.
//!
//! This module defines the `Tensor` struct and associated methods for managing tensor objects
//! using a specified backend. It also includes implementations for the `Source` and `Shape` traits,
//! enabling tensor operations and shape management.

use as_any::Downcast;
use std::sync::Arc;
use tengu_backend::{Backend, Linker, StorageType};

use crate::expression::{Shape, Source};
use crate::probe::Probe;
use crate::{Error, Result};

/// A tensor structure that holds data and metadata for tensor operations.
///
/// The `Tensor` struct is parameterized by a storage type `T` and a backend `B`. It includes
/// fields for the tensor's shape, backend, and the underlying tensor data.
pub struct Tensor<T: StorageType, B: Backend + 'static> {
    count: usize,
    shape: Vec<usize>,
    backend: Arc<B>,
    raw: Arc<B::Tensor<T>>,
}

impl<T: StorageType, B: Backend> Tensor<T, B> {
    /// Creates a new tensor with the specified backend, shape, and data.
    ///
    /// # Parameters
    /// - `backend`: A reference-counted backend object.
    /// - `shape`: A vector representing the shape of the tensor.
    /// - `count`: The number of elements in the tensor.
    /// - `tensor`: The underlying backend tensor.
    ///
    /// # Returns
    /// A new `Tensor` instance.
    pub fn new(backend: &Arc<B>, shape: Vec<usize>, count: usize, tensor: B::Tensor<T>) -> Self {
        Self {
            backend: Arc::clone(backend),
            count,
            shape,
            raw: tensor.into(),
        }
    }

    /// Returns a reference to the underlying backend tensor.
    ///
    /// # Returns
    /// A reference to the backend tensor.
    pub fn raw(&self) -> &B::Tensor<T> {
        &self.raw
    }

    /// Returns a probe object for inspecting the tensor data.
    ///
    /// # Returns
    /// A `Probe` object for the tensor.
    pub fn probe(&self) -> Probe<T, B> {
        use tengu_backend::Tensor;
        Probe::new(self.raw.probe())
    }

    /// Returns the label of the tensor.
    ///
    /// # Returns
    /// A string slice representing the tensor's label.
    pub fn label(&self) -> &str {
        use tengu_backend::Tensor;
        self.raw.label()
    }
}

// NOTE: Node

impl<T: StorageType, B: Backend> Source<B> for Tensor<T, B> {
    /// Checks if the tensor matches the shape of another tensor.
    ///
    /// # Parameters
    /// - `other`: Another source to compare against.
    ///
    /// # Returns
    /// A result indicating whether the shapes match.
    fn matches_to(&self, other: &dyn Source<B>) -> Result<bool> {
        let other = other.downcast_ref::<Self>().ok_or_else(|| Error::TypeMismatch)?;
        Ok(self.shape() == other.shape())
    }

    /// Copies the data from this tensor to another tensor using the provided linker.
    ///
    /// # Parameters
    /// - `to`: The target tesnro to link to.
    /// - `linker`: The linker to use for copying the link.
    ///
    /// # Returns
    /// A result indicating the success of the operation.
    fn copy(&self, to: &dyn Source<B>, linker: &mut B::Linker<'_>) -> Result<()> {
        let to = to.downcast_ref::<Self>().ok_or_else(|| Error::TypeMismatch)?;
        linker.copy_link(self.raw(), to.raw());
        Ok(())
    }
}

// NOTE: Shape

impl<T: StorageType, B: Backend> Shape for Tensor<T, B> {
    /// Returns the shape of the tensor.
    ///
    /// # Returns
    /// A slice representing the shape of the tensor.
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the number of elements in the tensor.
    ///
    /// # Returns
    /// The number of elements in the tensor.
    fn count(&self) -> usize {
        self.count
    }
}

// NOTE: Cloning

impl<T: StorageType, B: Backend> Clone for Tensor<T, B> {
    /// Clones the tensor, creating a new instance with the same data and metadata.
    ///
    /// # Returns
    /// A new `Tensor` instance that is a clone of the original.
    fn clone(&self) -> Self {
        Self {
            count: self.count,
            shape: self.shape.clone(),
            backend: Arc::clone(&self.backend),
            raw: Arc::clone(&self.raw),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::builder::LABEL_LENGTH;
    use crate::expression::Shape;
    use crate::Tengu;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn tensor_shape() {
        let tengu = Tengu::wgpu().await.unwrap();
        let tensor = tengu.tensor([3, 3, 3]).zero::<i32>();
        assert_eq!(tensor.count(), 27);
        assert_eq!(tensor.shape(), &[3, 3, 3]);
    }

    #[tokio::test]
    async fn tensor_label() {
        let tengu = Tengu::wgpu().await.unwrap();
        let tensor = tengu.tensor([3, 3, 3]).zero::<i32>();
        let label = tensor.label().unwrap();
        assert_eq!(label.len(), LABEL_LENGTH);
        assert!(label.chars().all(|c| c.is_alphabetic()));

        let tensor = tengu.tensor([3]).init(&[1, 2, 3]);
        let label = tensor.label().unwrap();
        assert_eq!(label.len(), LABEL_LENGTH);
        assert!(label.chars().all(|c| c.is_alphabetic()));
    }
}
