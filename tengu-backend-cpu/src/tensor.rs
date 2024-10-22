use std::cell::RefCell;

use tengu_backend_tensor::StorageType;

use crate::source::{AsSource, Source, Unsupported};

pub struct Tensor<T> {
    label: String,
    count: usize,
    shape: Vec<usize>,
    data: RefCell<Vec<T>>,
}

impl<T: StorageType> Tensor<T> {
    pub fn new(label: impl Into<String>, shape: impl Into<Vec<usize>>, data: impl Into<Vec<T>>) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            label: label.into(),
            count,
            shape,
            data: data.into().into(),
        }
    }

    pub fn copy_to(&self, other: &Self) {
        other.data.borrow_mut().copy_from_slice(&self.data.borrow());
    }

    pub fn empty(label: impl Into<String>, shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            label: label.into(),
            count,
            shape,
            data: vec![T::default(); count].into(),
        }
    }
}

// NOTE: AsSource implementations.

impl<T: StorageType> AsSource<Unsupported> for Tensor<T> {
    fn as_source(&self) -> Source<'_> {
        panic!("unsupported type");
    }
}

impl AsSource for Tensor<bool> {
    fn as_source(&self) -> crate::source::Source<'_> {
        Source::Bool(self)
    }
}

impl AsSource for Tensor<u32> {
    fn as_source(&self) -> crate::source::Source<'_> {
        Source::U32(self)
    }
}

impl AsSource for Tensor<i32> {
    fn as_source(&self) -> crate::source::Source<'_> {
        Source::I32(self)
    }
}

impl AsSource for Tensor<f32> {
    fn as_source(&self) -> crate::source::Source<'_> {
        Source::F32(self)
    }
}

// NOTE: Backend tensor trait implementation.

impl<T: StorageType> tengu_backend_tensor::Tensor for Tensor<T> {
    type Elem = T;

    fn label(&self) -> &str {
        &self.label
    }

    fn count(&self) -> usize {
        self.count
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    async fn retrieve(&self) -> anyhow::Result<std::borrow::Cow<'_, [<Self::Elem as StorageType>::IOType]>> {
        todo!()
    }
}
