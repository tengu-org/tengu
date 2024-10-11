//! This module defines the `Backend` struct which implements the `Backend` trait from the `tengu_backend` crate.
//!
//! The `Backend` struct is responsible for managing the WGPU device and providing methods to create and manipulate GPU resources
//! such as tensors, compute passes, and linkers. The `Backend` struct serves as the main entry point for managing GPU resources and
//! executing GPU operations. It provides methods to create tensors, perform compute operations, propagate data, and read out data
//! from the GPU.

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

/// The `Backend` struct is responsible for managing the WGPU device and providing methods to create and manipulate GPU resources.
pub struct Backend {
    device: Device,
}

impl Backend {
    /// Returns a reference to the `Device`.
    ///
    /// # Returns
    /// A reference to the `Device` object.
    pub(crate) fn device(&self) -> &Device {
        &self.device
    }
}

impl tengu_backend::Backend for Backend {
    type Tensor<T: StorageType> = WGPUTensor<T>;
    type Compute<'a> = WGPUCompute<'a>;
    type Processor<'a> = WGPUProcessor<'a>;
    type Linker<'a> = WGPULinker<'a>;
    type Readout<'a> = WGPUReadout<'a>;

    /// Creates a new `Backend` instance asynchronously.
    ///
    /// # Returns
    /// A result containing a reference-counted `Backend` instance or an error.
    async fn new() -> Result<Rc<Self>> {
        let device = WGPU::default_context().await.map_err(|e| Error::wgpu_error(e.into()))?;
        Ok(Rc::new(Self { device }))
    }

    /// Creates a new `Processor` instance.
    ///
    /// # Returns
    /// A new `Processor` instance.
    fn processor(&self) -> Self::Processor<'_> {
        WGPUProcessor::new()
    }

    /// Propagates data using the provided linker function.
    ///
    /// # Parameters
    /// - `call`: A function that takes a `Linker` and performs data propagation.
    fn propagate(&self, call: impl FnOnce(Self::Linker<'_>)) {
        let mut encoder = self.device.encoder("linker");
        call(WGPULinker::new(&mut encoder));
        self.device.submit(encoder.finish());
    }

    /// Executes a compute pass using the provided compute function.
    ///
    /// # Parameters
    /// - `label`: A label for the compute operations.
    /// - `call`: A function that takes a `Compute` and performs compute operations.
    fn compute(&self, label: &str, call: impl FnOnce(Self::Compute<'_>)) {
        let commands = self
            .device
            .encoder(label)
            .pass(label, |pass| call(WGPUCompute::new(&self.device, label, pass)))
            .finish();
        self.device.submit(commands);
    }

    /// Reads out data using the provided readout function.
    ///
    /// # Parameters
    /// - `label`: A label for the readout operations.
    /// - `call`: A function that takes a `Readout` and performs data readout from probes.
    fn readout(&self, label: &str, call: impl FnOnce(Self::Readout<'_>)) {
        let commands = self
            .device
            .encoder(label)
            .readout(|encoder| call(WGPUReadout::new(encoder)))
            .finish();
        self.device.submit(commands);
    }

    /// Creates a new tensor with the provided data.
    ///
    /// # Parameters
    /// - `label`: A label for the tensor.
    /// - `data`: A slice of data to initialize the tensor with.
    ///
    /// # Returns
    /// A new tensor initialized with the provided data.
    fn tensor<T: IOType>(self: &Rc<Self>, label: impl Into<String>, data: &[T]) -> Self::Tensor<T> {
        let buffer = self.device().buffer::<T>(BufferUsage::Read).with_data(data);
        WGPUTensor::new(self, label, data.len(), buffer)
    }

    /// Creates a new zero-initialized tensor with the specified count.
    ///
    /// # Parameters
    /// - `label`: A label for the tensor.
    /// - `count`: The number of elements in the tensor.
    ///
    /// # Returns
    /// A new zero-initialized tensor.
    fn zero<T: StorageType>(self: &Rc<Self>, label: impl Into<String>, count: usize) -> Self::Tensor<T> {
        let size = count.of::<T>();
        let buffer = self.device().buffer::<T>(BufferUsage::ReadWrite).empty(size);
        WGPUTensor::new(self, label, count, buffer)
    }

    /// Creates a probe for the specified count.
    ///
    /// # Parameters
    /// - `count`: The number of elements the probe will hold in the staging buffer.
    ///
    /// # Returns
    /// A new probe for the specified number of elements.
    fn probe<T: StorageType>(self: &Rc<Self>, count: usize) -> <Self::Tensor<T> as Tensor<T>>::Probe {
        let size = count.of::<T>();
        let buffer = self.device.buffer::<T>(BufferUsage::Staging).empty(size);
        Probe::new(self, buffer)
    }
}
