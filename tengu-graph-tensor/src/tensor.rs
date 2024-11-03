//! This module defines the `Tensor` struct and associated methods for managing tensor objects
//! using a specified backend. It also includes implementations for the `Source` and `Shape` traits,
//! enabling tensor operations and shape management.

use std::cell::OnceCell;
use std::rc::Rc;

use tengu_backend::Backend;
use tengu_tensor::StorageType;
use tengu_tensor::Tensor as RawTensor;

use crate::channel::Channel;
use crate::probe::Probe;
use crate::{Error, Result};

/// A tensor structure that holds data and metadata for tensor operations.
///
/// The `Tensor` struct is parameterized by a storage type `T` and a backend `B`. It includes
/// fields for the tensor's shape, backend, and the underlying tensor data.
pub struct Tensor<T: StorageType, B: Backend> {
    backend: Rc<B>,
    raw: Rc<B::Tensor<T>>,
    channel: OnceCell<Channel<T>>,
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
    pub fn new(backend: &Rc<B>, tensor: B::Tensor<T>) -> Self {
        Self {
            backend: Rc::clone(backend),
            raw: tensor.into(),
            channel: OnceCell::new(),
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
    pub fn probe(&self) -> Probe<T> {
        Probe::new(self.channel().receiver())
    }

    /// Returns the label of the tensor.
    ///
    /// # Returns
    /// A string slice representing the tensor's label.
    pub fn label(&self) -> Option<&str> {
        self.raw.label()
    }

    /// Returns the shape of the tensor.
    ///
    /// # Returns
    /// A slice of tensor's dimensions.
    pub fn shape(&self) -> &[usize] {
        self.raw.shape()
    }

    /// Returns the number of elements in the tensor.
    ///
    /// # Returns
    /// The number of elements in this tensor.
    pub fn count(&self) -> usize {
        self.raw.count()
    }

    /// Reads the tensor data from the source and sends it to associated probes.
    /// If the channel is full then there is no point wasting time on reading the data out - the
    /// previous message hasn't been read out by the probe yet. In this case the method will return
    /// immediately without retrieving and sending anything.
    ///
    /// # Returns
    /// A result indicating the success of the operation.
    pub async fn retrieve(&self) -> Result<()> {
        if self.channel().is_full() {
            return Ok(());
        }
        let data: Vec<_> = self.raw().retrieve().await.map_err(Error::ChannelError)?.into_owned();
        self.channel()
            .send(data)
            .await
            .map_err(|e| Error::ChannelError(e.into()))
    }

    /// Returns a reference to the channel for the tensor.
    ///
    /// # Returns
    /// A reference to the tensor's channel.
    fn channel(&self) -> &Channel<T> {
        self.channel.get_or_init(|| Channel::new())
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
            backend: Rc::clone(&self.backend),
            raw: Rc::clone(&self.raw),
            channel: self.channel.clone(),
        }
    }
}
