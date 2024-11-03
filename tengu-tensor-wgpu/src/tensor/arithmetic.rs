use std::ops::{Add, Div, Mul, Sub};

use tengu_tensor::{Arithmetic, StorageType, Unify};

use super::Tensor;

impl<T: StorageType> Arithmetic for Tensor<T>
where
    T: Add<Output = T> + Sub<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    fn add(&self, other: &Self) -> Self {
        let shape = self.shape.unify(&other.shape).expect("Shape mismatch");
        let expression = format!("({} + {})", self.expression, other.expression);
        Tensor::of_shape(&self.device, shape, expression)
    }

    fn sub(&self, other: &Self) -> Self {
        let shape = self.shape.unify(&other.shape).expect("Shape mismatch");
        let expression = format!("({} - {})", self.expression, other.expression);
        Tensor::of_shape(&self.device, shape, expression)
    }

    fn mul(&self, other: &Self) -> Self {
        let shape = self.shape.unify(&other.shape).expect("Shape mismatch");
        let expression = format!("({} * {})", self.expression, other.expression);
        Tensor::of_shape(&self.device, shape, expression)
    }

    fn div(&self, other: &Self) -> Self {
        let shape = self.shape.unify(&other.shape).expect("Shape mismatch");
        let expression = format!("({} / {})", self.expression, other.expression);
        Tensor::of_shape(&self.device, shape, expression)
    }
}
