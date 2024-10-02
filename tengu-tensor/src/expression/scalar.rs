use crate::shape::Shape;
use crate::StorageType;

#[derive(Clone)]
pub struct Scalar<T> {
    value: T,
    shape: Shape,
}

impl<T: StorageType> Scalar<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            shape: [1].into(),
        }
    }
}
