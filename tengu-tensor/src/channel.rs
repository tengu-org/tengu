use flume::{Receiver, Sender};
use tengu_backend::Backend;
use tengu_tensor_traits::StorageType;

use crate::{Error, Result};

pub type Payload<T, B> = Vec<<<<B as Backend>::Tensor<T> as tengu_tensor_traits::Tensor>::Elem as StorageType>::IOType>;

pub struct Channel<T: StorageType, B: Backend> {
    sender: Sender<Payload<T, B>>,
    receiver: Receiver<Payload<T, B>>,
}

impl<T: StorageType, B: Backend> Channel<T, B> {
    pub fn new() -> Self {
        let (sender, receiver) = flume::bounded(1);
        Self { sender, receiver }
    }

    pub fn is_full(&self) -> bool {
        self.sender.is_full()
    }

    pub fn receiver(&self) -> Receiver<Payload<T, B>> {
        self.receiver.clone()
    }

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
