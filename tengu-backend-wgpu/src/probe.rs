//! Probe implementation for the WGPU backend. This module defines the `Probe` struct which implements
//! the `Probe` trait from the `tengu_backend` crate. It is used for inspecting the data of a tensor in the WGPU backend.

use flume::Receiver;
use std::borrow::Cow;
use tengu_backend::{Error, IOType, Result};

/// The `Probe` struct is used for inspecting the data of a tensor in the WGPU backend.
pub struct Probe<T> {
    receiver: Receiver<Vec<T>>,
}

impl<T> Probe<T> {
    /// Creates a new `Probe` instance.
    ///
    /// # Parameters
    /// - `receiver`: A receiver for the tensor data.
    ///
    /// # Returns
    /// A new instance of `Probe`.
    pub fn new(receiver: Receiver<Vec<T>>) -> Self {
        Self { receiver }
    }
}

impl<T: IOType> tengu_backend::Probe<T> for Probe<T> {
    /// Asynchronously retrieves the data from the probe.
    ///
    /// # Returns
    /// A reference or an owned copy of the retrieved data if there are no errors. Otherwise,
    /// an error is returned.
    async fn retrieve(&self) -> Result<Cow<'_, [T]>> {
        let buffer = self.receiver.recv_async().await.map_err(|e| Error::OSError(e.into()))?;
        Ok(buffer.into())
    }
}
