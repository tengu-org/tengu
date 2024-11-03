//! This module defines the `Backend` struct which implements the `Backend` trait from the `tengu_backend` crate.
//! The `Backend` struct is responsible for managing the WGPU device and providing methods to create and manipulate GPU resources
//! such as tensors, compute passes, and linkers. The `Backend` struct serves as the main entry point for managing GPU resources and
//! executing GPU operations. It provides methods to create tensors, perform compute operations, propagate data, and read out data
//! from the GPU.

use std::rc::Rc;

use tengu_backend::{Error, Result};
use tengu_tensor::{IOType, StorageType};
use tengu_tensor_wgpu::Tensor;
use tengu_utils::Label;
use tengu_wgpu::{BufferUsage, ByteSize, Device, WGPU};
use tracing::trace;

use crate::limits::Limits;
use crate::operation::compute::Compute;
use crate::operation::propagate::Propagate;
use crate::operation::readout::Readout;

/// The `Backend` struct is responsible for managing the WGPU device and providing methods to create and manipulate GPU resources.
pub struct Backend {
    device: Rc<Device>,
}

impl Backend {
    /// Returns a reference to the `Device`.
    ///
    /// # Returns
    /// A reference to the `Device` object.
    pub(crate) fn device(&self) -> &Rc<Device> {
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
        Rc::new(Self {
            device: Rc::new(device),
        })
    }
}

// NOTE: tengu_backend::Backend implementation

impl tengu_backend::Backend for Backend {
    type Tensor<T: StorageType> = Tensor<T>;
    type Limits = Limits;

    // NOTE: Operations

    type Compute = Compute;
    type Propagate = Propagate;
    type Readout = Readout;

    /// Creates a new `Backend` instance asynchronously.
    ///
    /// # Returns
    /// A result containing a reference-counted `Backend` instance or an error.
    async fn new() -> Result<Rc<Self>> {
        let device = WGPU::default_context().await.map_err(|e| Error::WGPUError(e.into()))?;
        trace!("Created WGPU instance for backend");
        Ok(Rc::new(Self {
            device: Rc::new(device),
        }))
    }

    /// Returns the limits of the backend.
    ///
    /// # Returns
    /// The limits of the backend.
    fn limits(&self) -> Self::Limits {
        Limits::new(self)
    }

    /// Creates a new tensor with the provided data.
    ///
    /// # Parameters
    /// - `label`: A label for the tensor.
    /// - `shape`: The shape of the tensor.
    /// - `data`: A slice of data to initialize the tensor with.
    ///
    /// # Returns
    /// A new tensor initialized with the provided data.
    fn tensor<T: IOType>(
        self: &Rc<Self>,
        label: impl Into<Label>,
        shape: impl Into<Vec<usize>>,
        data: &[T],
    ) -> Self::Tensor<T> {
        let label = label.into();
        trace!("Creating new tensor '{label}'");
        let buffer = self
            .device()
            .buffer::<T>(label.value(), BufferUsage::Read)
            .with_data(data);
        Tensor::new(&self.device, label, shape, buffer)
    }

    /// Creates a new zero-initialized tensor with the specified shape.
    ///
    /// # Parameters
    /// - `label`: A label for the tensor.
    /// - `shape`: The shape of the tensor.
    ///
    /// # Returns
    /// A new zero-initialized tensor.
    fn zero<T: StorageType>(self: &Rc<Self>, label: impl Into<Label>, shape: impl Into<Vec<usize>>) -> Self::Tensor<T> {
        let label = label.into();
        let shape = shape.into();
        let size = shape.iter().product::<usize>().of::<T>();
        trace!("Creating new zero tensor '{label}'");
        let buffer = self
            .device()
            .buffer::<T>(label.value(), BufferUsage::ReadWrite)
            .empty(size);
        Tensor::new(&self.device, label, shape, buffer)
    }
}
