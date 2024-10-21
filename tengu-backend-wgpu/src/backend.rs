//! This module defines the `Backend` struct which implements the `Backend` trait from the `tengu_backend` crate.
//! The `Backend` struct is responsible for managing the WGPU device and providing methods to create and manipulate GPU resources
//! such as tensors, compute passes, and linkers. The `Backend` struct serves as the main entry point for managing GPU resources and
//! executing GPU operations. It provides methods to create tensors, perform compute operations, propagate data, and read out data
//! from the GPU.

use std::collections::HashSet;
use std::rc::Rc;

use tengu_backend::{Error, Result};
use tengu_backend_tensor::{IOType, StorageType};
use tengu_wgpu::{BufferUsage, ByteSize, Device, WGPU};
use tracing::trace;

use crate::compute::Compute as WGPUCompute;
use crate::limits::Limits as WGPULimits;
use crate::linker::Linker as WGPULinker;
use crate::processor::Processor as WGPUProcessor;
use crate::readout::Readout as WGPUReadout;
use crate::tensor::Tensor as WGPUTensor;

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

impl Backend {
    /// Creates a new `Backend` instance with the provided `Device`.
    ///
    /// # Parameters
    /// - `device`: The `Device` object to use for GPU operations.
    ///
    /// # Returns
    /// A new instance of `Backend`.
    pub fn from_device(device: Device) -> Rc<Self> {
        Rc::new(Self { device })
    }
}

// NOTE: tengu_backend::Backend implementation

impl tengu_backend::Backend for Backend {
    type Tensor<T: StorageType> = WGPUTensor<T>;
    type Compute<'a> = WGPUCompute<'a>;
    type Processor<'a> = WGPUProcessor<'a>;
    type Linker<'a> = WGPULinker<'a>;
    type Readout<'a> = WGPUReadout<'a>;
    type Limits = WGPULimits;

    /// Creates a new `Backend` instance asynchronously.
    ///
    /// # Returns
    /// A result containing a reference-counted `Backend` instance or an error.
    async fn new() -> Result<Rc<Self>> {
        let device = WGPU::default_context().await.map_err(|e| Error::WGPUError(e.into()))?;
        trace!("Created WGPU instance for backend");
        Ok(Rc::new(Self { device }))
    }

    /// Returns the limits of the backend.
    ///
    /// # Returns
    /// The limits of the backend.
    fn limits(&self) -> Self::Limits {
        WGPULimits::new(self)
    }

    /// Creates a new `Processor` instance.
    ///
    /// # Returns
    /// A new `Processor` instance.
    fn processor<'a>(&self, readouts: &'a HashSet<String>) -> Self::Processor<'a> {
        WGPUProcessor::new(readouts)
    }

    /// Propagates data using the provided linker function.
    ///
    /// # Parameters
    /// - `call`: A function that takes a `Linker` and performs data propagation.
    fn propagate(&self, call: impl FnOnce(Self::Linker<'_>)) {
        let mut encoder = self.device.encoder("linker");
        trace!("Executing propagation step");
        call(WGPULinker::new(&mut encoder));
        trace!("Submitting propagation commands to the queue");
        self.device.submit(encoder.finish());
    }

    /// Executes a compute pass using the provided compute function.
    ///
    /// # Parameters
    /// - `label`: A label for compute operations.
    /// - `call`: A function that takes a `Compute` and performs compute operations.
    fn compute<F>(&self, label: &str, call: F) -> Result<()>
    where
        F: FnOnce(Self::Compute<'_>) -> anyhow::Result<()>,
    {
        trace!("Executing compute step");
        let commands = self
            .device
            .encoder(label)
            .pass(label, |pass| call(WGPUCompute::new(&self.device, label, pass)))
            .map_err(|e| Error::ComputeError(e.into()))?
            .finish();
        trace!("Submitting compute commands to the queue");
        self.device.submit(commands);
        Ok(())
    }

    /// Copies data to staging buffers using the provided staging function.
    ///
    /// # Parameters
    /// - `label`: A label for the readout operations.
    /// - `call`: A function that takes a `Readout` value and copies data into staging buffers.
    fn readout(&self, label: &str, call: impl FnOnce(Self::Readout<'_>)) {
        trace!("Executing readout step");
        let commands = self
            .device
            .encoder(label)
            .stage(|encoder| call(WGPUReadout::new(encoder)))
            .finish();
        trace!("Submitting readout commands to the queue");
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
    fn tensor<T: IOType>(
        self: &Rc<Self>,
        label: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        data: &[T],
    ) -> Self::Tensor<T> {
        let label = label.into();
        trace!("Creating new tensor '{label}'");
        let buffer = self.device().buffer::<T>(&label, BufferUsage::Read).with_data(data);
        WGPUTensor::new(self, label, shape, buffer)
    }

    /// Creates a new zero-initialized tensor with the specified count.
    ///
    /// # Parameters
    /// - `label`: A label for the tensor.
    /// - `count`: The number of elements in the tensor.
    ///
    /// # Returns
    /// A new zero-initialized tensor.
    fn zero<T: StorageType>(
        self: &Rc<Self>,
        label: impl Into<String>,
        shape: impl Into<Vec<usize>>,
    ) -> Self::Tensor<T> {
        let label = label.into();
        let shape = shape.into();
        let size = shape.iter().product::<usize>().of::<T>();
        trace!("Creating new zero tensor '{label}'");
        let buffer = self.device().buffer::<T>(&label, BufferUsage::ReadWrite).empty(size);
        WGPUTensor::new(self, label, shape, buffer)
    }
}
