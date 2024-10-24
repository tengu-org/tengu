use tengu_backend_tensor::StorageType;

use super::Tensor;
use crate::cast::Cast;

// NOTE: Tensor casting.

macro_rules! impl_from {
    ( $type:ty ) => {
        impl<T: StorageType> From<&Tensor<$type>> for Tensor<T>
        where
            $type: Cast<T>,
        {
            fn from(other: &Tensor<$type>) -> Self {
                let data: Vec<_> = other.data.borrow().iter().map(|v| (*v).cast()).collect();
                Self::new(other.label.clone(), other.shape.clone(), data)
            }
        }
    };
}

impl_from!(u32);
impl_from!(i32);
impl_from!(f32);
impl_from!(bool);
