use flume::{Receiver, Sender};
use std::sync::Arc;
use tengu_backend::{IOType, Result};

use crate::Backend;

pub struct Stage<T> {
    backend: Arc<Backend>,
    buffer: Vec<T>,
    sender: Sender<Vec<T>>,
    receiver: Receiver<Vec<T>>,
}

impl<T: IOType> Stage<T> {
    pub fn new(backend: &Arc<Backend>, count: usize) -> Self {
        let (sender, receiver) = flume::bounded(1);
        Self {
            backend: Arc::clone(backend),
            buffer: vec![Default::default(); count],
            sender,
            receiver,
        }
    }

    pub fn buffer(&mut self) -> &mut [T] {
        &mut self.buffer
    }

    pub fn receiver(&self) -> Receiver<Vec<T>> {
        self.receiver.clone()
    }

    pub async fn readout(&self) -> Result<()> {
        Ok(())
    }
}
