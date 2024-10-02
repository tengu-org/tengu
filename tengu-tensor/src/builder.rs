use random_string::charsets::ALPHA;
use std::rc::Rc;

use crate::StorageType;
use crate::{expression::Expression, tensor::Tensor};
use tengu_backend::{Backend, IOType};

pub const LABEL_LENGTH: usize = 6;

pub struct Builder<B: Backend> {
    shape: Vec<usize>,
    count: usize,
    label: Option<String>,
    backend: Rc<B>,
}

impl<B: Backend> Builder<B> {
    pub fn new(backend: &Rc<B>, shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            shape,
            count,
            label: None,
            backend: Rc::clone(backend),
        }
    }

    pub fn label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn zero<T: StorageType>(mut self) -> Expression<T, B> {
        let label = self.get_or_create_label();
        let tensor = self.backend.zero(label, self.count);
        let tensor = Tensor::new(&self.backend, self.shape, self.count, tensor);
        Expression::Tensor(tensor)
    }

    pub fn init<T: IOType>(mut self, data: &[T]) -> Expression<T, B> {
        assert_eq!(data.len(), self.count, "data length does not match shape");
        let label = self.get_or_create_label();
        let tensor = self.backend.tensor(label, data);
        let tensor = Tensor::new(&self.backend, self.shape, self.count, tensor);
        Expression::Tensor(tensor)
    }

    pub fn bools(mut self, data: &[bool]) -> Expression<u32, B> {
        assert_eq!(data.len(), self.count, "data length does not match shape");
        let data = data.iter().map(|v| *v as u32).collect::<Vec<_>>();
        let label = self.get_or_create_label();
        let tensor = self.backend.tensor(label, &data);
        let tensor = Tensor::new(&self.backend, self.shape, self.count, tensor);
        Expression::Tensor(tensor)
    }

    fn get_or_create_label(&mut self) -> String {
        self.label
            .take()
            .unwrap_or_else(|| random_string::generate(LABEL_LENGTH, ALPHA))
    }
}
