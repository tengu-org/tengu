use std::{cell::OnceCell, rc::Rc};
use tengu_backend::{Backend, StorageType};
use tengu_wgpu::{Buffer, Encoder};

use crate::probe::Probe;
use crate::source::Source;
use crate::Backend as WGPUBackend;

pub struct Tensor<T: StorageType> {
    backend: Rc<WGPUBackend>,
    label: String,
    count: usize,
    probe: OnceCell<Probe<T::IOType>>,
    buffer: Rc<Buffer>,
}

impl<T: StorageType> Tensor<T> {
    pub fn new(backend: &Rc<WGPUBackend>, label: impl Into<String>, count: usize, buffer: Buffer) -> Self {
        Self {
            backend: Rc::clone(backend),
            label: label.into(),
            count,
            probe: OnceCell::new(),
            buffer: buffer.into(),
        }
    }
}

// Source implementation.

impl<T: StorageType> Source for Tensor<T> {
    fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    fn readout(&self, encoder: &mut Encoder) {
        if let Some(probe) = self.probe.get() {
            encoder.copy_buffer(&self.buffer, probe.buffer());
        }
    }

    fn count(&self) -> usize {
        self.count
    }

    fn label(&self) -> &str {
        &self.label
    }
}

// Raw tensor implementation.

impl<T: StorageType> tengu_backend::Tensor<T> for Tensor<T> {
    type Probe = Probe<T::IOType>;

    fn label(&self) -> &str {
        &self.label
    }

    fn probe(&self) -> &Self::Probe {
        self.probe.get_or_init(|| self.backend.probe::<T>(self.count))
    }
}

#[cfg(test)]
mod tests {
    use crate::Backend as WGPUBackend;
    use pretty_assertions::assert_eq;
    use tengu_backend::{Backend, Tensor};

    #[tokio::test]
    async fn tensor_emit() {
        let backend = WGPUBackend::new().await.unwrap();
        let tensor = backend.zero::<u32>("tenzor", 6);
        assert_eq!(tensor.label(), "tenzor");
    }
}
