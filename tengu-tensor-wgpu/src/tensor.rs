//! This module provides the implementation of the `Tensor` struct, which represents a tensor on the WGPU backend.
//! It includes functionality for creating tensors, managing their data, and interfacing with the GPU for compute operations.

use std::borrow::Cow;
use std::cell::OnceCell;
use std::marker::PhantomData;
use std::rc::Rc;

use tengu_backend::Error;
use tengu_tensor::StorageType;
use tengu_tensor::Tensor as RawTensor;
use tengu_utils::Label;
use tengu_wgpu::Device;
use tengu_wgpu::{Buffer, BufferUsage, ByteSize, Encoder};

use crate::Kind;

mod arithmetic;
mod cast;
mod copy_from;
mod relational;
mod unary_fn;

/// Represents a tensor on the WGPU backend.
pub struct Tensor<T> {
    device: Rc<Device>,
    label: Option<Label>,
    count: usize,
    shape: Vec<usize>,
    kind: Kind,
    expression: String,
    staging_buffer: Rc<OnceCell<Buffer>>,
    buffer: Rc<OnceCell<Buffer>>,
    phantom: PhantomData<T>,
}

impl<T: StorageType> Tensor<T> {
    /// Creates a new `Tensor` with the specified backend, label, shape, and buffer.
    ///
    /// # Parameters
    /// - `backend`: A reference-counted pointer to the WGPU backend.
    /// - `label`: A string label for identifying the tensor.
    /// - `shape`: The shape of the tensor as a vector of unsigned integers.
    /// - `buffer`: The GPU buffer storing the tensor's data.
    ///
    /// # Returns
    /// A new instance of `Tensor`.
    pub fn new(device: &Rc<Device>, label: impl Into<Label>, shape: impl Into<Vec<usize>>, buffer: Buffer) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        let label = label.into();
        let expression = label.to_string();
        Self {
            device: Rc::clone(device),
            label: Some(label),
            count,
            shape,
            kind: Kind::Generic,
            expression,
            staging_buffer: Rc::new(OnceCell::new()),
            buffer: Rc::new(buffer.into()),
            phantom: PhantomData,
        }
    }

    pub fn like<S: StorageType>(other: &Tensor<S>, expression: String) -> Self {
        Self {
            device: Rc::clone(&other.device),
            label: None,
            count: other.count,
            shape: other.shape.clone(),
            kind: other.kind,
            expression,
            staging_buffer: Rc::new(OnceCell::new()),
            buffer: Rc::new(OnceCell::new()),
            phantom: PhantomData,
        }
    }

    pub fn of_shape(device: &Rc<Device>, shape: impl Into<Vec<usize>>, expression: String) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            device: Rc::clone(device),
            label: None,
            count,
            shape,
            kind: Kind::Generic,
            expression,
            staging_buffer: Rc::new(OnceCell::new()),
            buffer: Rc::new(OnceCell::new()),
            phantom: PhantomData,
        }
    }

    pub fn scalar(device: &Rc<Device>, value: T) -> Self {
        Self {
            device: Rc::clone(device),
            label: None,
            count: 1,
            shape: vec![1],
            kind: Kind::Scalar,
            expression: value.to_string(),
            staging_buffer: Rc::new(OnceCell::new()),
            buffer: Rc::new(OnceCell::new()),
            phantom: PhantomData,
        }
    }

    pub fn emit(&self) -> &str {
        &self.expression
    }

    pub fn declaration(&self, group: usize, binding: usize) -> String {
        let label = self.label().expect("input tensors should have a label");
        let access = access(self.buffer().usage());
        let ty = std::any::type_name::<T>();
        format!("@group({group}) @binding({binding}) var<storage, {access}> {label}: array<{ty}>;")
    }

    /// Copies the tensor's data from the GPU buffer to the staging buffer using the provided encoder.
    ///
    /// # Parameters
    /// - `encoder`: The encoder used to copy the buffer data.
    pub fn readout(&self, encoder: &mut Encoder) {
        encoder.copy_buffer(self.buffer(), self.stage());
    }

    /// Returns a reference to the GPU buffer storing the tensor's data.
    ///
    /// # Returns
    /// A reference to the `Buffer`.
    pub fn buffer(&self) -> &Buffer {
        self.buffer
            .get()
            .expect("intermediate vectors don't have associated buffers")
    }

    /// Returns a reference to the tensor's staging object, initializing it if necessary.
    ///
    /// # Returns
    /// A reference to the tensor's staging object.
    fn stage(&self) -> &Buffer {
        self.staging_buffer.get_or_init(|| {
            let size = self.count.of::<T>();
            let label = self.label.as_ref().map(|label| label.value()).unwrap_or_default();
            self.device.buffer::<T>(label, BufferUsage::Staging).empty(size)
        })
    }
}

/// Determines the access type for a buffer based on its usage.
///
/// # Parameters
/// - `usage`: The buffer usage type.
///
/// # Returns
/// A static string representing the access type.
fn access(usage: BufferUsage) -> &'static str {
    match usage {
        BufferUsage::Read => "read",
        BufferUsage::Write => "write",
        BufferUsage::ReadWrite => "read_write",
        BufferUsage::Staging => panic!("cannot declare a staging buffer in a shader"),
    }
}

// NOTE: Tensor trait implementation.

impl<T: StorageType> RawTensor<T> for Tensor<T> {
    /// Returns the label of the tensor.
    ///
    /// # Returns
    /// The label of the tensor.
    fn label(&self) -> Option<&str> {
        self.label.as_ref().map(|label| label.value())
    }

    /// Returns the number of elements in the tensor.
    ///
    /// # Returns
    /// The number of elements in the tensor.
    fn count(&self) -> usize {
        self.count
    }

    /// Returns the shape of the tensor.
    ///
    /// # Returns
    /// The shape of the tensor as a slice of unsigned integers.
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Retrieves staging buffer data from the GPU memory into the CPU buffer.
    ///
    /// # Returns
    /// A `Cow` containing either a reference or owned buffer with the tensor data.
    async fn retrieve(&self) -> anyhow::Result<Cow<'_, [T::IOType]>> {
        let staging_buffer = self.stage();
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::wait()).panic_on_timeout();
        receiver
            .recv_async()
            .await
            .map_err(|e| Error::WGPUError(e.into()))?
            .map_err(|e| Error::WGPUError(e.into()))?;
        let data = buffer_slice.get_mapped_range();
        let buffer = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();
        Ok(buffer.into())
    }
}

// NOTE: Clone implementation.

impl<T: StorageType> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            device: Rc::clone(&self.device),
            label: self.label.clone(),
            count: self.count,
            shape: self.shape.clone(),
            kind: self.kind,
            expression: self.expression.clone(),
            staging_buffer: self.staging_buffer.clone(),
            buffer: self.buffer.clone(),
            phantom: self.phantom,
        }
    }
}
