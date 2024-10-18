use std::cell::RefCell;
use std::sync::Arc;
use tengu_backend::StorageType;

use crate::probe::Probe;
use crate::source::Source;
use crate::stage::Stage;
use crate::Backend as CPUBackend;

pub struct Tensor<T: StorageType> {
    backend: Arc<CPUBackend>,
    label: String,
    count: usize,
    buffer: RefCell<Vec<T>>,
    stage: RefCell<Option<Stage<T::IOType>>>,
}

impl<T: StorageType> Tensor<T> {
    pub fn new(backend: &Arc<CPUBackend>, label: String, count: usize) -> Self {
        Self {
            backend: Arc::clone(backend),
            label,
            count,
            buffer: RefCell::new(vec![T::default(); count]),
            stage: RefCell::new(None),
        }
    }

    pub fn with_data(backend: &Arc<CPUBackend>, label: String, data: &[T]) -> Self {
        Self {
            backend: Arc::clone(backend),
            label,
            count: data.len(),
            buffer: RefCell::new(data.to_vec()),
            stage: RefCell::new(None),
        }
    }

    pub fn copy_from(&self, other: &Self) {
        let other_buffer = other.buffer.borrow();
        self.buffer.borrow_mut().copy_from_slice(&other_buffer);
    }
}

// NOTE: Source implementation.

impl<T: StorageType> Source for Tensor<T> {
    fn readout(&self) {
        if let Some(stage) = self.stage.borrow_mut().as_mut() {
            for (i, v) in self.buffer.borrow().iter().enumerate() {
                stage.buffer()[i] = v.convert();
            }
        }
    }

    fn count(&self) -> usize {
        self.count
    }

    fn label(&self) -> &str {
        &self.label
    }
}

// NOTE: Raw tensor implementation.

impl<T: StorageType> tengu_backend::Tensor<T> for Tensor<T> {
    type Probe = Probe<T::IOType>;

    fn label(&self) -> &str {
        &self.label
    }

    fn probe(&self) -> Self::Probe {
        let mut stage = self.stage.borrow_mut();
        let stage = stage.get_or_insert_with(|| Stage::new(&self.backend, self.count));
        Probe::new(stage.receiver())
    }
}

#[cfg(test)]
mod tests {
    use crate::Backend as CPUBackend;
    use pretty_assertions::assert_eq;
    use tengu_backend::{Backend, Tensor};

    #[tokio::test]
    async fn tensor_emit() {
        let backend = CPUBackend::new().await.unwrap();
        let tensor = backend.zero::<u32>("tenzor", 6);
        assert_eq!(tensor.label(), "tenzor");
    }
}
