use tengu_tensor::StorageType;
use tengu_tensor_cpu::{PrimitiveCast, Tensor};

use super::Source;

impl<'a> Source<'a> {
    pub fn cast<T>(&self) -> Self
    where
        T: StorageType,
        u32: PrimitiveCast<T>,
        i32: PrimitiveCast<T>,
        f32: PrimitiveCast<T>,
        bool: PrimitiveCast<T>,
    {
        match self {
            Source::U32(_) => Tensor::<T>::from(self.as_ref::<u32>()).into(),
            Source::I32(_) => Tensor::<T>::from(self.as_ref::<i32>()).into(),
            Source::F32(_) => Tensor::<T>::from(self.as_ref::<f32>()).into(),
            Source::Bool(_) => Tensor::<T>::from(self.as_ref::<f32>()).into(),
        }
    }
}
