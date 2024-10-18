use flume::Receiver;
use std::borrow::Cow;
use tengu_backend::{Error, IOType, Result};

pub struct Probe<T> {
    receiver: Receiver<Vec<T>>,
}

impl<T> Probe<T> {
    pub fn new(receiver: Receiver<Vec<T>>) -> Self {
        Self { receiver }
    }
}

impl<T: IOType> tengu_backend::Probe<T> for Probe<T> {
    async fn retrieve(&self) -> Result<Cow<'_, [T]>> {
        let buffer = self.receiver.recv_async().await.map_err(|e| Error::OSError(e.into()))?;
        Ok(buffer.into())
    }
}
