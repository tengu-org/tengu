use random_string::charsets::ALPHA;
use std::{cell::OnceCell, marker::PhantomData, rc::Rc};
use tengu_wgpu::{Buffer, BufferUsage, ByteSize, Encoder};

use crate::{Expression, Probe, Tengu, WGSLType};

const LABEL_LENGTH: usize = 6;

// Tensor implementation

pub struct Tensor<T> {
    label: String,
    buffer: Rc<Buffer>,
    count: usize,
    shape: Vec<usize>,
    probe: OnceCell<Probe>,
    tengu: Rc<Tengu>,
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
        format!("@group({group}) @binding({binding}) var<storage, {access}> {label}: array<{type_name}>;")
    }
}

// Cloning

impl<T> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            label: self.label.clone(),
            buffer: Rc::clone(&self.buffer),
            count: self.count,
            shape: self.shape.clone(),
            tengu: Rc::clone(&self.tengu),
            probe: OnceCell::new(),
            phantom: PhantomData,
        }
    }
}

// Builder implementation

pub struct TensorBuilder {
    shape: Vec<usize>,
    count: usize,
    label: Option<String>,
    tengu: Rc<Tengu>,
}

impl TensorBuilder {
    pub fn new(tengu: &Rc<Tengu>, shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            shape,
            count,
            label: None,
            tengu: Rc::clone(tengu),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn empty<T: WGSLType>(mut self) -> Expression<T> {
        let size = self.count.of::<T>();
        let buffer = self.tengu.device().buffer::<T>(BufferUsage::ReadWrite).empty(size);
        let tensor = Tensor {
            label: self.label(),
            buffer: buffer.into(),
            count: self.count,
            shape: self.shape,
            tengu: self.tengu,
            probe: OnceCell::new(),
            phantom: PhantomData,
        };
        Expression::Tensor(tensor)
    }

    pub fn init<T: WGSLType>(mut self, data: &[T]) -> Expression<T> {
        assert_eq!(data.len(), self.count, "data length does not match shape");
        let buffer = self.tengu.device().buffer::<T>(BufferUsage::Read).with_data(data);
        let tensor = Tensor {
            label: self.label(),
            buffer: buffer.into(),
            count: self.count,
            shape: self.shape,
            tengu: self.tengu,
            probe: OnceCell::new(),
            phantom: PhantomData,
        };
        Expression::Tensor(tensor)
    }

    fn label(&mut self) -> String {
        self.label
            .take()
            .unwrap_or_else(|| random_string::generate(LABEL_LENGTH, ALPHA))
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
        let tensor = tengu.tensor([3, 3, 3]).empty::<i32>();
        assert_eq!(tensor.count(), 27);
        assert_eq!(tensor.shape(), &[3, 3, 3]);
    }

    #[tokio::test]
    async fn tensor_label() {
        let tengu = Tengu::new().await.unwrap();
        let tensor = tengu.tensor([3, 3, 3]).empty::<i32>();
        assert_eq!(tensor.label().len(), LABEL_LENGTH);
        assert!(tensor.label().chars().all(|c| c.is_alphabetic()));

        let tensor = tengu.tensor([3]).init(&[1, 2, 3]);
        assert_eq!(tensor.label().len(), LABEL_LENGTH);
        assert!(tensor.label().chars().all(|c| c.is_alphabetic()));
    }

    #[tokio::test]
    async fn tensor_emit() {
        let tengu = Tengu::new().await.unwrap();
        let tensor = tengu.tensor([3, 3, 3]).with_label("tenzor").empty::<i32>();
        assert_eq!(tensor.emit(), "tenzor[idx]");

        let tensor = tengu.tensor([3]).with_label("tenzor").init(&[1, 2, 3]);
        assert_eq!(tensor.label().len(), LABEL_LENGTH);
        assert!(tensor.label().chars().all(|c| c.is_alphabetic()));
    }
}
