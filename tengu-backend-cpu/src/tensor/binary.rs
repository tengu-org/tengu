use std::ops::{Add, Div, Mul, Sub};

use tengu_backend_tensor::StorageType;

use super::Tensor;

impl<T: StorageType + Add<Output = T>> Add for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        let data: Vec<T> = self
            .data
            .borrow()
            .iter()
            .zip(rhs.data.borrow().iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Tensor::new(self.label.clone(), self.shape.clone(), data)
    }
}

impl<T: StorageType + Sub<Output = T>> Sub for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        let data: Vec<T> = self
            .data
            .borrow()
            .iter()
            .zip(rhs.data.borrow().iter())
            .map(|(a, b)| *a - *b)
            .collect();
        Tensor::new(self.label.clone(), self.shape.clone(), data)
    }
}

impl<T: StorageType + Mul<Output = T>> Mul for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        let data: Vec<T> = self
            .data
            .borrow()
            .iter()
            .zip(rhs.data.borrow().iter())
            .map(|(a, b)| *a * *b)
            .collect();
        Tensor::new(self.label.clone(), self.shape.clone(), data)
    }
}

impl<T: StorageType + Div<Output = T>> Div for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, rhs: Self) -> Self::Output {
        let data: Vec<T> = self
            .data
            .borrow()
            .iter()
            .zip(rhs.data.borrow().iter())
            .map(|(a, b)| *a / *b)
            .collect();
        Tensor::new(self.label.clone(), self.shape.clone(), data)
    }
}

impl<T: StorageType + PartialEq> PartialEq for &Tensor<T> {
    fn eq(&self, other: &Self) -> bool {
        let lhs = self.data.borrow();
        let rhs = other.data.borrow();
        *lhs == *rhs
    }
}
