use std::ops::{Add, Div, Mul, Sub};

use tengu_tensor::{Arithmetic, StorageType, Unify};

use super::Tensor;

impl<T: StorageType> Arithmetic for Tensor<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    fn add(&self, other: &Self) -> Self {
        let shape = self.shape.unify(&other.shape).expect("Shape mismatch");
        let data: Vec<T> = self
            .data
            .borrow()
            .iter()
            .zip(other.data.borrow().iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Tensor::from_shape(shape, data)
    }

    fn sub(&self, other: &Self) -> Self {
        let shape = self.shape.unify(&other.shape).expect("Shape mismatch");
        let data: Vec<T> = self
            .data
            .borrow()
            .iter()
            .zip(other.data.borrow().iter())
            .map(|(a, b)| *a - *b)
            .collect();
        Tensor::from_shape(shape, data)
    }

    fn mul(&self, other: &Self) -> Self {
        let shape = self.shape.unify(&other.shape).expect("Shape mismatch");
        let data: Vec<T> = self
            .data
            .borrow()
            .iter()
            .zip(other.data.borrow().iter())
            .map(|(a, b)| *a * *b)
            .collect();
        Tensor::from_shape(shape, data)
    }

    fn div(&self, other: &Self) -> Self {
        let shape = self.shape.unify(&other.shape).expect("Shape mismatch");
        let data: Vec<T> = self
            .data
            .borrow()
            .iter()
            .zip(other.data.borrow().iter())
            .map(|(a, b)| *a / *b)
            .collect();
        Tensor::from_shape(shape, data)
    }
}
