use tengu_tensor::StorageType;

use super::Source;
use crate::{cast::Cast, tensor::Tensor};

impl<'a> Source<'a> {
    pub fn cast<T>(&self) -> Self
    where
        T: StorageType,
        u32: Cast<T>,
        i32: Cast<T>,
        f32: Cast<T>,
        bool: Cast<T>,
    {
        match self {
            Source::U32(_) => Tensor::<T>::from(self.as_ref::<u32>()).into(),
            Source::I32(_) => Tensor::<T>::from(self.as_ref::<i32>()).into(),
            Source::F32(_) => Tensor::<T>::from(self.as_ref::<f32>()).into(),
            Source::Bool(_) => Tensor::<T>::from(self.as_ref::<f32>()).into(),
        }
    }
}
