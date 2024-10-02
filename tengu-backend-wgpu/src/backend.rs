use std::rc::Rc;
use tengu_backend::{IOType, Result, StorageType, Tensor};
use tengu_wgpu::{BufferUsage, ByteSize, Device, WGPU};

use crate::compute::Compute as WGPUCompute;
use crate::linker::Linker as WGPULinker;
use crate::probe::Probe;
use crate::processor::Processor as WGPUProcessor;
use crate::readout::Readout as WGPUReadout;
use crate::tensor::Tensor as WGPUTensor;
use crate::Error;

pub struct Backend {
    device: Device,
}

impl Backend {
    pub(crate) fn device(&self) -> &Device {
        &self.device
    }
}

impl tengu_backend::Backend for Backend {
    type Tensor<T: StorageType> = WGPUTensor<T>;
    type Compute<'a> = WGPUCompute<'a>;
    type Processor<'a> = WGPUProcessor<'a>;
    type Linker = WGPULinker;
    type Readout<'a> = WGPUReadout<'a>;

    async fn new() -> Result<Rc<Self>> {
        let device = WGPU::default_context().await.map_err(|e| Error::wgpu_error(e.into()))?;
        Ok(Rc::new(Self { device }))
    }

    fn processor(&self) -> Self::Processor<'_> {
        WGPUProcessor::new()
    }

    fn linker(&self) -> Self::Linker {
        let encoder = self.device.encoder("linker");
        WGPULinker::new(encoder)
    }

    fn compute(&self, label: &str, call: impl FnOnce(Self::Compute<'_>)) {
        let commands = self
            .device
            .encoder(label)
            .pass(label, |pass| call(WGPUCompute::new(&self.device, label, pass)))
            .finish();
        self.device.submit(commands);
    }

    fn readout(&self, label: &str, call: impl FnOnce(Self::Readout<'_>)) {
        let commands = self
            .device
            .encoder(label)
            .readout(|encoder| call(WGPUReadout::new(encoder)))
            .finish();
        self.device.submit(commands);
    }

    fn tensor<T: IOType>(self: &Rc<Self>, label: impl Into<String>, data: &[T]) -> Self::Tensor<T> {
        let buffer = self.device().buffer::<T>(BufferUsage::Read).with_data(data);
        WGPUTensor::new(self, label, data.len(), buffer)
    }

    fn zero<T: StorageType>(self: &Rc<Self>, label: impl Into<String>, count: usize) -> Self::Tensor<T> {
        let size = count.of::<T>();
        let buffer = self.device().buffer::<T>(BufferUsage::ReadWrite).empty(size);
        WGPUTensor::new(self, label, count, buffer)
    }

    fn probe<T: StorageType>(self: &Rc<Self>, count: usize) -> <Self::Tensor<T> as Tensor<T>>::Probe {
        let size = count.of::<T>();
        let buffer = self.device.buffer::<T>(BufferUsage::Staging).empty(size);
        Probe::new(self, buffer)
    }
}
