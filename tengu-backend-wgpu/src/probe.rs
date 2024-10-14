//! This module defines the `Probe` struct which is used for retrieving data from a GPU buffer in an asynchronous manner.
//! It utilizes the WGPU backend and integrates with the `tengu_backend` crate to provide efficient data retrieval capabilities.

use std::{marker::PhantomData, rc::Rc};

use tengu_backend::{Error, IOType, Result};
use tengu_wgpu::Buffer;

use crate::Backend;

/// The `Probe` struct is used to manage and retrieve data from a GPU buffer.
/// It holds a reference-counted handle to the staging buffer. During the readout operation it will
/// be used as the destination target of the copy from the tensor associated with this probe.
pub struct Probe<T> {
    backend: Rc<Backend>,
    buffer: Buffer,
    phantom: PhantomData<T>,
}

impl<T> Probe<T> {
    /// Creates a new `Probe` instance.
    ///
    /// # Parameters
    /// - `backend`: A reference-counted handle to the backend.
    /// - `buffer`: The stating buffer from which data will be retrieved.
    ///
    /// # Returns
    /// A new instance of `Probe`.
    pub fn new(backend: &Rc<Backend>, buffer: Buffer) -> Self {
        Self {
            backend: Rc::clone(backend),
            buffer,
            phantom: PhantomData,
        }
    }

    /// Returns a reference to the GPU buffer.
    ///
    /// # Returns
    /// A reference to the probe's staging buffer.
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl<T: IOType> tengu_backend::Probe<T> for Probe<T> {
    /// Asynchronously retrieves data from the GPU buffer and stores it into the provided vector.
    ///
    /// # Parameters
    /// - `buffer`: A mutable reference to a vector where the retrieved data will be stored.
    ///
    /// # Returns
    /// A result indicating success or failure of the operation.
    async fn retrieve_to(&self, buffer: &mut Vec<T>) -> Result<()> {
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
        *buffer = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.buffer.unmap();
        Ok(())
    }
}
