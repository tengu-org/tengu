use std::{borrow::Cow, cell::RefCell};

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

    pub fn repeat(label: impl Into<String>, shape: impl Into<Vec<usize>>, elem: T) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            label: label.into(),
            count,
            shape,
            data: vec![elem; count].into(),
        }
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

    pub fn copy_to(&self, other: &Self) {
        other.data.borrow_mut().copy_from_slice(&self.data.borrow());
    }
}

// NOTE: AsSource implementations.

impl<T: StorageType> AsSource<Unsupported> for Tensor<T> {
    fn as_source(&self) -> Source<'_> {
        panic!("unsupported type");
    }
}

macro_rules! impl_as_source {
    ( $type:ty ) => {
        impl AsSource for Tensor<$type> {
            fn as_source(&self) -> Source<'_> {
                self.into()
            }
        }
    };
}

impl_as_source!(bool);
impl_as_source!(u32);
impl_as_source!(i32);
impl_as_source!(f32);

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

    async fn retrieve(&self) -> anyhow::Result<Cow<'_, [<Self::Elem as StorageType>::IOType]>> {
        todo!()
    }
}

// NOTE: Clone implementation.

impl<T: StorageType> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            label: self.label.clone(),
            count: self.count,
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }
}
