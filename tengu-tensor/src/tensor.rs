use std::rc::Rc;

use crate::expression::{Shape, Source};
use crate::probe::Probe;
use crate::{Error, Result};
use as_any::Downcast;
use tengu_backend::{Backend, Linker, StorageType};

pub struct Tensor<T: StorageType, B: Backend + 'static> {
    count: usize,
    shape: Vec<usize>,
    backend: Rc<B>,
    tensor: Rc<B::Tensor<T>>,
}

impl<T: StorageType, B: Backend> Tensor<T, B> {
    pub fn new(backend: &Rc<B>, shape: Vec<usize>, count: usize, tensor: B::Tensor<T>) -> Self {
        Self {
            backend: Rc::clone(backend),
            count,
            shape,
            tensor: tensor.into(),
        }
    }

    pub fn raw_tensor(&self) -> &B::Tensor<T> {
        &self.tensor
    }

    pub fn probe(&self) -> Probe<'_, T, B> {
        use tengu_backend::Tensor;
        let raw_probe = self.tensor.probe();
        Probe::new(raw_probe, self.count)
    }

    pub fn label(&self) -> &str {
        use tengu_backend::Tensor;
        self.tensor.label()
    }
}

// Node

impl<T: StorageType, B: Backend> Source<B> for Tensor<T, B> {
    fn matches_with(&self, other: &dyn Source<B>) -> Result<bool> {
        let other = other.downcast_ref::<Self>().ok_or_else(|| Error::TypeMismatch)?;
        Ok(self.shape() == other.shape())
    }

    fn copy_link(&self, to: &dyn Source<B>, linker: &mut B::Linker) -> Result<()> {
        let to = to.downcast_ref::<Self>().ok_or_else(|| Error::TypeMismatch)?;
        linker.copy_link(self.raw_tensor(), to.raw_tensor());
        Ok(())
    }
}

impl<T: StorageType, B: Backend> Shape for Tensor<T, B> {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn count(&self) -> usize {
        self.count
    }
}

// Cloning

impl<T: StorageType, B: Backend> Clone for Tensor<T, B> {
    fn clone(&self) -> Self {
        Self {
            count: self.count,
            shape: self.shape.clone(),
            backend: Rc::clone(&self.backend),
            tensor: Rc::clone(&self.tensor),
        }
    }
}

// Tests

#[cfg(test)]
mod tests {
    use crate::builder::LABEL_LENGTH;
    use crate::expression::Shape;
    use crate::Tengu;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn tensor_shape() {
        let tengu = Tengu::wgpu().await.unwrap();
        let tensor = tengu.tensor([3, 3, 3]).zero::<i32>();
        assert_eq!(tensor.count(), 27);
        assert_eq!(tensor.shape(), &[3, 3, 3]);
    }

    #[tokio::test]
    async fn tensor_label() {
        let tengu = Tengu::wgpu().await.unwrap();
        let tensor = tengu.tensor([3, 3, 3]).zero::<i32>();
        let label = tensor.label().unwrap();
        assert_eq!(label.len(), LABEL_LENGTH);
        assert!(label.chars().all(|c| c.is_alphabetic()));

        let tensor = tengu.tensor([3]).init(&[1, 2, 3]);
        let label = tensor.label().unwrap();
        assert_eq!(label.len(), LABEL_LENGTH);
        assert!(label.chars().all(|c| c.is_alphabetic()));
    }
}
