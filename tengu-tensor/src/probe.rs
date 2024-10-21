//! Module for probing tensor values in the Tengu tensor computation framework.
//!
//! This module defines the `Probe` struct and associated methods for inspecting and retrieving
//! values from tensors. It provides functionalities to turn probing on and off and to retrieve
//! tensor data asynchronously.

use flume::Receiver;
use tengu_backend::Backend;
use tengu_tensor_traits::StorageType;

use crate::channel::Payload;
use crate::{Error, Result};

/// A struct for probing tensor values.
///
/// The `Probe` struct holds to store recently retrieved values. Since retrieval operations is
/// asyncronous and time-consuming, we use this cache to allow accessing retrieved values
/// synchronously.
pub struct Probe<T: StorageType, B: Backend> {
    receiver: Receiver<Payload<T, B>>,
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
    pub fn new(receiver: Receiver<Payload<T, B>>) -> Self {
        Self { receiver }
    }

    /// Asynchronously retrieves tensor values into the inner buffer.
    ///
    /// # Returns
    /// A reference or an owned copy of the retrieved data if there are no errors. Otherwise,
    /// an error is returned.
    pub async fn retrieve(&self) -> Result<Payload<T, B>> {
        self.receiver
            .recv_async()
            .await
            .map_err(|e| Error::ChannelError(e.into()))
    }
}
