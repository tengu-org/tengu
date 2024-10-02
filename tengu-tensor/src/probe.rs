use std::rc::Rc;

use tengu_wgpu::{Buffer, BufferUsage, ByteSize};

use crate::frontend::Shape;
use crate::tensor::Tensor;
use crate::{Error, IOType, Result, Tengu};

pub struct Probe {
    buffer: Buffer,
    tengu: Rc<Tengu>,
    on: bool,
}

impl Probe {
    pub fn new<T>(tengu: &Rc<Tengu>, from: &Tensor<T>) -> Self {
        let size = from.count().of::<T>();
        let buffer = tengu.device().buffer::<T>(BufferUsage::Staging).empty(size);
        Self {
            tengu: Rc::clone(tengu),
            buffer,
            on: true,
        }
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn turn_off(&mut self) {
        self.on = false;
    }

    pub fn turn_on(&mut self) {
        self.on = true;
    }

    pub async fn retrieve<T: IOType>(&self) -> Result<Vec<T>> {
        if !self.on {
            return Ok(Vec::new());
        }
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
