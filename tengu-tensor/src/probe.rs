use std::marker::PhantomData;
use std::sync::Arc;

use tengu_wgpu::{Buffer, BufferUsage, ByteSize};

use crate::{Error, Result, Tengu};

pub trait Probable<T> {
    fn probe(&mut self) -> &Probe<T>;
}

// Probe implementation

pub struct Probe<T> {
    buffer: Buffer,
    tengu: Arc<Tengu>,
    phantom: PhantomData<T>,
}

impl<T> Probe<T> {
    pub fn new(tengu: &Arc<Tengu>, count: usize) -> Self {
        let size = count.bytes();
        let buffer = tengu.device().buffer::<T>(BufferUsage::Staging).empty(size);
        Self {
            tengu: Arc::clone(tengu),
            buffer,
            phantom: PhantomData,
        }
    }

    pub async fn retrieve(&self) -> Result<Vec<f32>> {
        let buffer_slice = self.buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.tengu.device().poll(wgpu::Maintain::wait()).panic_on_timeout();
        receiver
            .recv_async()
            .await
            .map_err(|e| Error::InternalError(e.into()))?
            .map_err(|e| Error::InternalError(e.into()))?;
        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.buffer.unmap();
        Ok(result)
    }
}
