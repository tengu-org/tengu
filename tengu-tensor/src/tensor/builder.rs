use super::Tensor;
use crate::{expression::Expression, PodType, Tengu, WGSLType};

use random_string::charsets::ALPHA;
use std::{cell::OnceCell, marker::PhantomData, rc::Rc};
use tengu_wgpu::{BufferUsage, ByteSize};

pub const LABEL_LENGTH: usize = 6;

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

    pub fn init<T: PodType>(mut self, data: &[T]) -> Expression<T> {
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

    pub fn bools(mut self, data: &[bool]) -> Expression<bool> {
        assert_eq!(data.len(), self.count, "data length does not match shape");
        let data = data.iter().map(|v| *v as u32).collect::<Vec<_>>();
        let buffer = self.tengu.device().buffer::<u32>(BufferUsage::Read).with_data(&data);
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
