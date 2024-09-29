use random_string::charsets::ALPHA;
use std::{cell::OnceCell, marker::PhantomData, ops::Add, sync::Arc};
use tengu_wgpu::{Buffer, BufferUsage, ByteSize, Encoder};

use crate::{Expression, Probe, Tengu};

const LABEL_LENGTH: usize = 6;

// Tensor implementation

pub struct Tensor<T> {
    label: String,
    buffer: Buffer,
    count: usize,
    shape: Vec<usize>,
    probe: OnceCell<Probe>,
    tengu: Arc<Tengu>,
    phantom: PhantomData<T>,
}

impl<T> Tensor<T> {
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn emit(&self) -> String {
        format!("{label}[idx]", label = self.label.clone())
    }

    pub fn probe(&self) -> &Probe {
        self.probe.get_or_init(|| Probe::new(&self.tengu, self))
    }

    pub fn read(&self, encoder: &mut Encoder) {
        if let Some(probe) = self.probe.get() {
            encoder.copy_buffer(self.buffer(), probe.buffer());
        }
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
}

// Builder implementation

pub struct TensorBuilder<T> {
    shape: Vec<usize>,
    count: usize,
    label: Option<String>,
    tengu: Arc<Tengu>,
    phantom: PhantomData<T>,
}

impl<T> TensorBuilder<T> {
    pub fn new(tengu: &Arc<Tengu>, shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            shape,
            count,
            label: None,
            tengu: Arc::clone(tengu),
            phantom: PhantomData,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn empty(mut self) -> Tensor<T> {
        let size = self.count.bytes();
        let buffer = self.tengu.device().buffer::<T>(BufferUsage::ReadWrite).empty(size);
        Tensor {
            label: self.label(),
            buffer,
            count: self.count,
            shape: self.shape,
            tengu: self.tengu,
            probe: OnceCell::new(),
            phantom: PhantomData,
        }
    }

    pub fn init(mut self, data: &[T]) -> Tensor<T>
    where
        T: bytemuck::Pod,
    {
        assert_eq!(data.len(), self.count, "data length does not match shape");
        let buffer = self.tengu.device().buffer::<T>(BufferUsage::ReadWrite).with_data(data);
        Tensor {
            label: self.label(),
            buffer,
            count: self.count,
            shape: self.shape,
            tengu: self.tengu,
            probe: OnceCell::new(),
            phantom: PhantomData,
        }
    }

    fn label(&mut self) -> String {
        self.label
            .take()
            .unwrap_or_else(|| random_string::generate(LABEL_LENGTH, ALPHA))
    }
}

// Operations

impl<T> Add for Tensor<T> {
    type Output = Expression<T>;

    fn add(self, other: Tensor<T>) -> Self::Output {
        assert_eq!(self.shape, other.shape, "tensor shapes should match");
        Expression::add(self, other)
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn tensor_builder() {
        let tengu = Tengu::new().await.unwrap();
        let tensor = tengu.tensor::<i32>([3, 3, 3]).empty();
        assert_eq!(tensor.count(), 27);
        assert_eq!(tensor.shape(), &[3, 3, 3]);
    }

    #[tokio::test]
    async fn tensor_declaration() {
        let tengu = Tengu::new().await.unwrap();
        let tensor = tengu.tensor::<i32>([3, 3, 3]).with_label("tz").empty();
        let declaration = tensor.declaration(1, 2);
        assert_eq!(
            declaration,
            "@group(1) @binding(2) var<storage, read_write> tz: array<i32>"
        );
    }

    #[tokio::test]
    async fn tensor_label() {
        let tengu = Tengu::new().await.unwrap();
        let tensor = tengu.tensor::<i32>([3, 3, 3]).empty();
        assert_eq!(tensor.label().len(), LABEL_LENGTH);
        assert!(tensor.label().chars().all(|c| c.is_alphabetic()));

        let tensor = tengu.tensor([3]).init(&[1, 2, 3]);
        assert_eq!(tensor.label().len(), LABEL_LENGTH);
        assert!(tensor.label().chars().all(|c| c.is_alphabetic()));
    }

    #[tokio::test]
    async fn tensor_emit() {
        let tengu = Tengu::new().await.unwrap();
        let tensor = tengu.tensor::<i32>([3, 3, 3]).with_label("tenzor").empty();
        assert_eq!(tensor.emit(), "tenzor[idx]");

        let tensor = tengu.tensor([3]).with_label("tenzor").init(&[1, 2, 3]);
        assert_eq!(tensor.label.len(), LABEL_LENGTH);
        assert!(tensor.label.chars().all(|c| c.is_alphabetic()));
    }

    #[tokio::test]
    #[should_panic]
    async fn test_shape_mismatch() {
        let tengu = Tengu::new().await.unwrap();
        let lhs = tengu.tensor::<i32>([1, 2, 3]).empty();
        let rhs = tengu.tensor::<i32>([3, 2, 1]).empty();
        let _ = lhs + rhs;
    }
}
