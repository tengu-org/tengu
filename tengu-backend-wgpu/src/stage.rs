//! This module defines the `Stage` struct which is used to retrieve data from a GPU buffer in an asynchronous manner.
//! It utilizes the WGPU backend and integrates with the `tengu_backend` crate to provide efficient data retrieval capabilities.

use flume::{Receiver, Sender};
use std::sync::Arc;
use tengu_backend::{Error, IOType, Result};
use tengu_wgpu::Buffer;

use crate::Backend;

/// The `Stage` struct is used to manage and retrieve data from a GPU buffer.
/// It holds handle to the staging buffer. During the readout operation it will
/// be used as the destination target of the copy from the tensor associated with this probe.
pub struct Stage<T> {
    backend: Arc<Backend>,
    buffer: Buffer,
    sender: Sender<Vec<T>>,
    receiver: Receiver<Vec<T>>,
}

impl<T: IOType> Stage<T> {
    /// Creates a new `Stage` instance.
    ///
    /// # Parameters
    /// - `backend`: A reference-counted handle to the backend.
    /// - `buffer`: The stating buffer to which the data will be copied during readout.
    ///
    /// # Returns
    /// A new instance of `Stage`.
    pub fn new(backend: &Arc<Backend>, buffer: Buffer) -> Self {
        let (sender, receiver) = flume::bounded(1);
        Self {
            backend: Arc::clone(backend),
            buffer,
            sender,
            receiver,
        }
    }

    /// Returns a reference to the GPU buffer.
    ///
    /// # Returns
    /// A reference to the probe's staging buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Returns a reference to the probe's receiver. During the retireval phase, `Stage` will
    /// send its data into the internal `Sender`. The probe on the other side of the channel will
    /// receive these data using this reciever.
    ///
    /// # Returns
    /// A reciever to be used by a probe.
    pub fn receiver(&self) -> Receiver<Vec<T>> {
        self.receiver.clone()
    }

    /// Asynchronously retrieves data from the GPU buffer and sends it through the internal
    /// channel.
    ///
    /// # Returns
    /// A result indicating success or failure of the operation.
    pub async fn retrieve(&self) -> Result<()> {
        let buffer_slice = self.buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.backend.device().poll(wgpu::Maintain::wait()).panic_on_timeout();
        receiver
            .recv_async()
            .await
            .map_err(|e| Error::WGPUError(e.into()))?
            .map_err(|e| Error::WGPUError(e.into()))?;
        let data = buffer_slice.get_mapped_range();
        let buffer = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.buffer.unmap();
        let _ = self.sender.try_send(buffer); // We don't care what's going on with receivers.
        Ok(())
    }
}
