//! Module for building tensors in the Tengu tensor computation framework.
//!
//! This module defines the `Builder` struct and associated methods for creating tensors with various initializations.
//! It provides an interface to set the shape, label, and initial values of tensors.

use random_string::charsets::ALPHA;
use std::rc::Rc;
use tengu_backend::{Backend, IOType};

use crate::expression::Expression;
use crate::tensor::Tensor;
use crate::StorageType;

/// The length of the label generated for tensors if no label is provided.
pub(crate) const LABEL_LENGTH: usize = 6;

/// A struct for building tensors with specified shapes and initializations.
///
/// The `Builder` struct provides methods to set the shape, label, and initial values of tensors.
pub struct Builder<B: Backend> {
    shape: Vec<usize>,
    count: usize,
    label: Option<String>,
    backend: Rc<B>,
}

impl<B: Backend> Builder<B> {
    /// Creates a new `Builder` instance with the specified backend and shape.
    ///
    /// # Parameters
    /// - `backend`: A reference-counted backend instance.
    /// - `shape`: The shape of the tensor.
    ///
    /// # Returns
    /// A new `Builder` instance.
    pub fn new(backend: &Rc<B>, shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            shape,
            count,
            label: None,
            backend: Rc::clone(backend),
        }
    }
    /// Sets the label for the tensor.
    ///
    /// # Parameters
    /// - `label`: The label for the tensor.
    ///
    /// # Returns
    /// The `Builder` instance with the updated label.
    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Creates a tensor initialized to zero.
    ///
    /// # Type Parameters
    /// - `T`: The storage type of the tensor.
    ///
    /// # Returns
    /// An expression representing the tensor initialized to zero.
    pub fn zero<T: StorageType>(mut self) -> Expression<T, B> {
        let label = self.get_or_create_label();
        let tensor = self.backend.zero(label, self.count);
        let tensor = Tensor::new(&self.backend, self.shape, self.count, tensor);
        Expression::Tensor(tensor)
    }

    /// Creates a tensor initialized with the specified data.
    ///
    /// # Type Parameters
    /// - `T`: The I/O type of the tensor.
    ///
    /// # Parameters
    /// - `data`: A slice of data to initialize the tensor.
    ///
    /// # Returns
    /// An expression representing the tensor initialized with the data.
    ///
    /// # Panics
    /// Panics if the length of the data does not match the shape of the tensor.
    pub fn init<T: IOType>(mut self, data: &[T]) -> Expression<T, B> {
        assert_eq!(data.len(), self.count, "data length does not match shape");
        let label = self.get_or_create_label();
        let tensor = self.backend.tensor(label, data);
        let tensor = Tensor::new(&self.backend, self.shape, self.count, tensor);
        Expression::Tensor(tensor)
    }

    /// Creates a tensor initialized with the specified boolean data. The resulting tensor will
    /// use u32 instead of bool because bool cannot be used as an IOType on some backends.
    ///
    /// # Parameters
    /// - `data`: A slice of boolean data to initialize the tensor.
    ///
    /// # Returns
    /// An expression representing the tensor initialized with the boolean data.
    ///
    /// # Panics
    /// Panics if the length of the data does not match the shape of the tensor.
    pub fn bools(mut self, data: &[bool]) -> Expression<u32, B> {
        assert_eq!(data.len(), self.count, "data length does not match shape");
        let data = data.iter().map(|v| *v as u32).collect::<Vec<_>>();
        let label = self.get_or_create_label();
        let tensor = self.backend.tensor(label, &data);
        let tensor = Tensor::new(&self.backend, self.shape, self.count, tensor);
        Expression::Tensor(tensor)
    }

    /// Retrieves or creates a new random label for the tensor.
    ///
    /// # Returns
    /// A string representing the label of the tensor.
    fn get_or_create_label(&mut self) -> String {
        self.label
            .take()
            .unwrap_or_else(|| random_string::generate(LABEL_LENGTH, ALPHA))
    }
}
