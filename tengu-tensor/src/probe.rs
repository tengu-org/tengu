//! Module for probing tensor values in the Tengu tensor computation framework.
//!
//! This module defines the `Probe` struct and associated methods for inspecting and retrieving
//! values from tensors. It provides functionalities to turn probing on and off and to retrieve
//! tensor data asynchronously.

use std::borrow::Cow;
use tengu_backend::{Backend, IOType, StorageType, Tensor};

use crate::Result;

/// A struct for probing tensor values.
///
/// The `Probe` struct holds to store recently retrieved values. Since retrieval operations is
/// asyncronous and time-consuming, we use this cache to allow accessing retrieved values
/// synchronoously.
pub struct Probe<T: StorageType, B: Backend> {
    raw: <B::Tensor<T> as Tensor<T>>::Probe,
    on: bool,
}

impl<T: StorageType, B: Backend> Probe<T, B> {
    /// Creates a new `Probe` instance.
    ///
    /// # Parameters
    /// - `probe`: A reference to the probe object.
    /// - `count`: The number of elements to initialize in the buffer.
    ///
    /// # Returns
    /// A new `Probe` instance.
    pub fn new(probe: <B::Tensor<T> as Tensor<T>>::Probe) -> Self {
        Self { raw: probe, on: true }
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

    /// Asynchronously retrieves tensor values into the inner buffer.
    ///
    /// # Returns
    pub async fn retrieve(&self) -> Result<Option<Cow<'_, [T::IOType]>>>
    where
        T: IOType,
    {
        use tengu_backend::Probe;
        let result = match self.on {
            true => Some(self.raw.retrieve().await?),
            false => None,
        };
        Ok(result)
    }
}
