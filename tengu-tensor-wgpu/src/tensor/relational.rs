use tengu_tensor::{Relational, StorageType, Unify};

use super::Tensor;

impl<T: StorageType + PartialEq> Relational for Tensor<T> {
    type Output = Tensor<bool>;

    fn eq(&self, other: &Self) -> Self::Output {
        let shape = self.shape.unify(&other.shape).expect("Shape mismatch");
        let expression = format!("({} == {})", self.expression, other.expression);
        Tensor::of_shape(&self.device, shape, expression)
    }

    fn neq(&self, other: &Self) -> Self::Output {
        let shape = self.shape.unify(&other.shape).expect("Shape mismatch");
        let expression = format!("({} != {})", self.expression, other.expression);
        Tensor::of_shape(&self.device, shape, expression)
    }
}
