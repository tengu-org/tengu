//! Module for building tensors in the Tengu tensor computation framework.
//!
//! This module defines the `Builder` struct and associated methods for creating tensors with various initializations.
//! It provides an interface to set the shape, label, and initial values of tensors.

use rand::distributions::uniform::SampleUniform;
use rand::distributions::Uniform;
use rand::Rng;
use rand_distr::Distribution;
use rand_distr::Normal;
use rand_distr::StandardNormal;
use random_string::charsets::ALPHA;
use tengu_tensor::Tensor;

use std::rc::Rc;

use num::Float;
use tengu_backend::Backend;
use tengu_backend_tensor::IOType;

use crate::expression::Expression;
use crate::{Error, Result, StorageType};

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
        let tensor = self.backend.zero(label, self.shape);
        let tensor = Tensor::new(&self.backend, tensor);
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
        let tensor = self.backend.tensor(label, self.shape, data);
        let tensor = Tensor::new(&self.backend, tensor);
        Expression::Tensor(tensor)
    }

    /// Creates a tensor initialized with random data. This is the most general method if you need
    /// to create a random tensor with specific Rng and distribution. You can use `uniform` or
    /// `normal` for most popular distributions.
    ///
    /// # Type Parameters
    /// - `T`: The I/O type of the tensor.
    /// - `D`: The distribution type for generating random data.
    /// - `R`: The random number generator type.
    ///
    /// # Parameters
    /// - `rng`: The random number generator instance.
    /// - `distr`: The distribution for generating random data.
    ///
    /// # Returns
    /// An expression representing the tensor initialized with random data.
    pub fn random<T, D, R>(mut self, rng: R, distr: D) -> Expression<T, B>
    where
        T: IOType,
        D: Distribution<T>,
        R: Rng,
    {
        let label = self.get_or_create_label();
        let data = rng.sample_iter(distr).take(self.count).collect::<Vec<_>>();
        let tensor = self.backend.tensor(label, self.shape, &data);
        let tensor = Tensor::new(&self.backend, tensor);
        Expression::Tensor(tensor)
    }

    /// Creates a tensor initialized with random data drawn from a uniform distribution.
    /// The data will be drawn from the range `[low, high)`.
    ///
    /// # Type Parameters
    /// - `T`: The I/O type of the tensor.
    ///
    /// # Parameters
    /// - `low`: The lower bound of the uniform distribution.
    /// - `high`: The upper bound of the uniform distribution.
    ///
    /// # Returns
    /// An expression representing the tensor initialized with random data drawn from a uniform distribution.
    pub fn uniform<T: IOType + SampleUniform>(self, low: T, high: T) -> Expression<T, B> {
        let rng = rand::thread_rng();
        self.random(rng, Uniform::new(low, high))
    }

    /// Creates a tensor initialized with random data drawn from a normal distribution.
    /// The data will be drawn from the distribution with the specified mean and standard deviation.
    ///
    /// # Type Parameters
    /// - `T`: The I/O type of the tensor.
    ///
    /// # Parameters
    /// - `mean`: The mean of the normal distribution.
    /// - `std_dev`: The standard deviation of the normal distribution.
    ///
    /// # Returns
    /// An expression representing the tensor initialized with random data drawn from a normal distribution.
    /// If the standard normal distribution is incorrent (e.g. is infinite), an error will be returned.
    pub fn normal<T>(self, mean: T, std_dev: T) -> Result<Expression<T, B>>
    where
        T: IOType + Float,
        StandardNormal: Distribution<T>,
    {
        let rng = rand::thread_rng();
        let distr = Normal::new(mean, std_dev).map_err(|e| Error::ParameterError(e.into()))?;
        Ok(self.random(rng, distr))
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
        let tensor = self.backend.tensor(&label, self.shape, &data);
        let tensor = Tensor::new(&self.backend, tensor);
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
