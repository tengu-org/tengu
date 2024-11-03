use tengu_tensor::{Relational, StorageType};

use super::Tensor;

impl<T: StorageType + PartialEq> Relational for Tensor<T> {
    type Output = Tensor<bool>;

    fn eq(&self, other: &Self) -> Self::Output {
        let lhs = self.data.borrow();
        let rhs = other.data.borrow();
        let data: Vec<_> = lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| *lhs == *rhs).collect();
        Tensor::from_tensor(self, data)
    }

    fn neq(&self, other: &Self) -> Self::Output {
        let lhs = self.data.borrow();
        let rhs = other.data.borrow();
        let data: Vec<_> = lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| *lhs != *rhs).collect();
        Tensor::from_tensor(self, data)
    }
}
