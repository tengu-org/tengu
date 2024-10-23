use tengu_backend_tensor::StorageType;

use super::Tensor;

macro_rules! impl_from {
    ( $type:ty ) => {
        impl<T: StorageType + From<$type>> From<&Tensor<$type>> for Tensor<T> {
            fn from(other: &Tensor<$type>) -> Self {
                let data: Vec<_> = other.data.borrow().iter().map(|v| T::from(*v)).collect();
                Self::new(other.label.clone(), other.shape.clone(), data)
            }
        }
    };
}

impl_from!(bool);
impl_from!(u32);
impl_from!(i32);
impl_from!(f32);
