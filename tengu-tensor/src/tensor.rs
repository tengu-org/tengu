use std::{cell::OnceCell, marker::PhantomData, rc::Rc};
use tengu_wgpu::{Buffer, BufferUsage};

use crate::backend::{Datum, Emit};
use crate::frontend::{Shape, Source};
use crate::{probe::Probe, Tengu};

pub use builder::Builder;

mod builder;

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

// Node

impl<T> Shape for Tensor<T> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn count(&self) -> usize {
        self.count
    }
}

impl<T> Emit for Tensor<T> {
    fn emit(&self) -> String {
        format!("{label}[idx]", label = self.label.clone())
    }
}

impl<T> Datum for Tensor<T> {
    fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    fn probe(&self) -> &Probe {
        self.probe.get_or_init(|| Probe::new(&self.tengu, self))
    }
}

impl<T> Source for Tensor<T> {
    fn label(&self) -> &str {
        &self.label
    }

    fn declaration(&self, group: usize, binding: usize) -> String {
        let label = &self.label;
        // let type_name = name_of::<T>();
        let type_name = std::any::type_name::<T>();
        let access = access(self.buffer().usage());
        format!("@group({group}) @binding({binding}) var<storage, {access}> {label}: array<{type_name}>;")
    }
}

// Helpers

fn access(usage: BufferUsage) -> &'static str {
    match usage {
        BufferUsage::Read => "read",
        BufferUsage::Write => "write",
        BufferUsage::ReadWrite => "read_write",
        BufferUsage::Staging => panic!("cannot declare a staging buffer in a shader"),
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

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use builder::LABEL_LENGTH;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn tensor_builder() {
        let tengu = Tengu::new().await.unwrap();
        let tensor = tengu.tensor([3, 3, 3]).zero::<i32>();
        assert_eq!(tensor.count(), 27);
        assert_eq!(tensor.shape(), &[3, 3, 3]);
    }

    #[tokio::test]
    async fn tensor_label() {
        let tengu = Tengu::new().await.unwrap();
        let tensor = tengu.tensor([3, 3, 3]).zero::<i32>();
        let label = tensor.label().unwrap();
        assert_eq!(label.len(), LABEL_LENGTH);
        assert!(label.chars().all(|c| c.is_alphabetic()));

        let tensor = tengu.tensor([3]).init(&[1, 2, 3]);
        let label = tensor.label().unwrap();
        assert_eq!(label.len(), LABEL_LENGTH);
        assert!(label.chars().all(|c| c.is_alphabetic()));
    }

    #[tokio::test]
    async fn tensor_emit() {
        let tengu = Tengu::new().await.unwrap();
        let tensor = tengu.tensor([3, 3, 3]).label("tenzor").zero::<i32>();
        assert_eq!(tensor.emit(), "tenzor[idx]");

        let tensor = tengu.tensor([3]).label("tenzor").init(&[1, 2, 3]);
        let label = tensor.label().unwrap();
        assert_eq!(label, "tenzor");
    }
}
