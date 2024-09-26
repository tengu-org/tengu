use random_string::charsets::ALPHA;
use std::{ops::Add, sync::Arc};

use tengu_wgpu::{Buffer, BufferUsage, ByteSize};

use crate::probe::impl_probable;
use crate::{Emit, Expression, Probable, Probe, Tengu};

pub struct TensorBuilder {
    shape: Vec<usize>,
    count: usize,
    tengu: Arc<Tengu>,
}

impl TensorBuilder {
    pub fn new(tengu: Arc<Tengu>, shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self { shape, count, tengu }
    }

    pub fn empty<T>(self) -> Tensor<T> {
        let size = self.count.bytes();
        let buffer = self.tengu.device().buffer::<T>(BufferUsage::ReadWrite).empty(size);
        Tensor {
            label: random_string::generate(6, ALPHA),
            buffer,
            count: self.count,
            shape: self.shape,
            tengu: self.tengu,
            probe: None,
        }
    }

    pub fn init<T>(self, data: &[T]) -> Tensor<T>
    where
        T: bytemuck::Pod,
    {
        assert_eq!(data.len(), self.count, "data length does not match shape");
        let buffer = self.tengu.device().buffer::<T>(BufferUsage::ReadWrite).with_data(data);
        Tensor {
            label: random_string::generate(6, ALPHA),
            buffer,
            count: self.count,
            shape: self.shape,
            tengu: self.tengu,
            probe: None,
        }
    }
}

pub struct Tensor<T> {
    label: String,
    buffer: Buffer,
    count: usize,
    shape: Vec<usize>,
    probe: Option<Probe<T>>,
    tengu: Arc<Tengu>,
}

impl<T> Tensor<T> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn declaration(&self, group: usize, binding: usize) -> String {
        let label = &self.label;
        let type_name = std::any::type_name::<T>();
        let access = match self.buffer.usage() {
            BufferUsage::Read => "read",
            BufferUsage::Write => "write",
            BufferUsage::ReadWrite => "read_write",
            BufferUsage::Staging => panic!("cannot declare a staging buffer in a shader"),
        };
        format!("@group({group}) @binding({binding}) var<storage, {access}> {label}: array<{type_name}>")
    }

    pub fn probe(&mut self) -> &Probe<T> {
        self.probe = Some(Probe::new(Arc::clone(&self.tengu), self.count));
        self.probe
            .as_ref()
            .expect("tensor probe should be non-empty after setting it")
    }
}

// Trait implementations

impl<T: 'static> Emit for Tensor<T> {
    fn emit(&self) -> String {
        format!("{label}[idx]", label = self.label.clone())
    }
}

impl_probable!(Tensor<T>);

// Operations

impl<T> Tensor<T> {
    pub fn add(self, other: Tensor<T>) -> Expression<T> {
        Expression::add(self, other)
    }
}

impl<T> Add for Tensor<T> {
    type Output = Expression<T>;

    fn add(self, other: Tensor<T>) -> Self::Output {
        self.add(other)
    }
}
