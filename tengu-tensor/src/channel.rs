//! Channel for communication between tensors and associated probes. This module provides
//! functionalities for sending and receiving tensor data between tensors and probes.

use flume::{Receiver, Sender};
use tengu_backend::Backend;
use tengu_backend_tensor::{StorageType, Tensor};

use crate::{Error, Result};

/// A type alias for the payload of the channel.
pub type Payload<T, B> = Vec<<<<B as Backend>::Tensor<T> as Tensor>::Elem as StorageType>::IOType>;

/// A struct for managing communication between tensors and probes.
pub struct Channel<T: StorageType, B: Backend> {
    sender: Sender<Payload<T, B>>,
    receiver: Receiver<Payload<T, B>>,
}

impl<T: StorageType, B: Backend> Channel<T, B> {
    /// Creates a new `Channel` instance.
    ///
    /// # Returns
    /// A new `Channel` instance.
    pub fn new() -> Self {
        let (sender, receiver) = flume::bounded(1);
        Self { sender, receiver }
    }

    /// Checks if the channel is full.
    ///
    /// # Returns
    /// A boolean indicating if the channel is full.
    pub fn is_full(&self) -> bool {
        self.sender.is_full()
    }

    /// Create a clone of reciever hold by this channel.
    ///
    /// # Returns
    /// A clone of the receiver.
    pub fn receiver(&self) -> Receiver<Payload<T, B>> {
        self.receiver.clone()
    }

    /// Sends tensor data to the channel.
    ///
    /// # Parameters
    /// - `data`: The tensor data to send.
    ///
    /// # Returns
    /// A result indicating success or failure.
    pub async fn send(&self, data: Payload<T, B>) -> Result<()> {
        self.sender
            .send_async(data)
            .await
            .map_err(|e| Error::ChannelError(e.into()))
    }
}

impl<T: StorageType, B: Backend> Default for Channel<T, B> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: StorageType, B: Backend> Clone for Channel<T, B> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            receiver: self.receiver.clone(),
        }
    }
}
