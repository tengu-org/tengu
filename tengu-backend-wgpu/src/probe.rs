use std::{marker::PhantomData, rc::Rc};

use tengu_backend::{Error, IOType, Result};
use tengu_wgpu::Buffer;

use crate::Backend;

pub struct Probe<T> {
    backend: Rc<Backend>,
    buffer: Buffer,
    phantom: PhantomData<T>,
}

impl<T> Probe<T> {
    pub fn new(backend: &Rc<Backend>, buffer: Buffer) -> Self {
        Self {
            backend: Rc::clone(backend),
            buffer,
            phantom: PhantomData,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

impl<T: IOType> tengu_backend::Probe<T> for Probe<T> {
    async fn retrieve_to(&self, buffer: &mut Vec<T>) -> Result<()> {
        let buffer_slice = self.buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.backend.device().poll(wgpu::Maintain::wait()).panic_on_timeout();
        receiver
            .recv_async()
            .await
            .map_err(|e| Error::WGPUBackendError(e.into()))?
            .map_err(|e| Error::WGPUBackendError(e.into()))?;
        let data = buffer_slice.get_mapped_range();
        *buffer = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.buffer.unmap();
        Ok(())
    }
}
