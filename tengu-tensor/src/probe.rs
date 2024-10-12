//! Module for probing tensor values in the Tengu tensor computation framework.
//!
//! This module defines the `Probe` struct and associated methods for inspecting and retrieving
//! values from tensors. It provides functionalities to turn probing on and off and to retrieve
//! tensor data asynchronously.

use num::Zero;
use tengu_backend::{Backend, IOType, StorageType, Tensor};

use crate::{Error, Result};

/// A struct for probing tensor values.
///
/// The `Probe` struct holds to store recently retrieved values. Since retrieval operations is
/// asyncronous and time-consuming, we use this cache to allow accessing retrieved values
/// synchronoously.
pub struct Probe<'a, T: StorageType, B: Backend> {
    probe: &'a <B::Tensor<T> as Tensor<T>>::Probe,
    buffer: Vec<T::IOType>,
    on: bool,
}

impl<'a, T: StorageType, B: Backend> Probe<'a, T, B>
where
    T::IOType: Zero,
{
    /// Creates a new `Probe` instance.
    ///
    /// # Parameters
    /// - `probe`: A reference to the probe object.
    /// - `count`: The number of elements to initialize in the buffer.
    ///
    /// # Returns
    /// A new `Probe` instance.
    pub fn new(probe: &'a <B::Tensor<T> as Tensor<T>>::Probe, count: usize) -> Self {
        Self {
            probe,
            buffer: vec![T::IOType::zero(); count],
            on: true,
        }
    }

    /// Turns off probing.
    ///
    /// When probing is turned off, the staging buffer will not be updated with new values.
    pub fn turn_off(&mut self) {
        self.on = false;
    }

    /// Turns on probing.
    ///
    /// When probing is turned on, the staging buffer will be updated with new values.
    pub fn turn_on(&mut self) {
        self.on = true;
    }

    /// Returns probe status.
    ///
    /// # Returns
    /// True if the probe is on, and false otherwise.
    pub fn is_on(&self) -> bool {
        self.on
    }

    /// Return cached results of data retrieved from a staging buffer.
    ///
    /// The data returned will be the one updated during the last retrieve call.
    ///
    /// # Returns
    /// A slices of `IOType` values.
    pub fn data(&self) -> &[T::IOType] {
        &self.buffer
    }

    /// Asynchronously retrieves tensor values into the inner buffer.
    ///
    /// # Returns
    /// A result containing a slice of the retrieved values or an error.
    /// Any subsequent calls to `data` will return the same values.
    pub async fn retrieve(&mut self) -> Result<&[T::IOType]>
    where
        T: IOType,
    {
        use tengu_backend::Probe;
        if !self.on {
            return Ok(&self.buffer);
        }
        self.probe
            .retrieve_to(&mut self.buffer)
            .await
            .map_err(Error::BackendError)?;
        Ok(&self.buffer)
    }
}
