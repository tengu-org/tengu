use tengu_backend_tensor::StorageType;

use super::Tensor;

impl<T: StorageType + PartialEq> Tensor<T> {
    pub fn eq(&self, other: &Self) -> Tensor<bool> {
        let lhs = self.data.borrow();
        let rhs = other.data.borrow();
        let data: Vec<_> = lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| *lhs == *rhs).collect();
        Tensor::new(self.label.clone(), self.shape.clone(), data)
    }

    pub fn neq(&self, other: &Self) -> Tensor<bool> {
        let lhs = self.data.borrow();
        let rhs = other.data.borrow();
        let data: Vec<_> = lhs.iter().zip(rhs.iter()).map(|(lhs, rhs)| *lhs != *rhs).collect();
        Tensor::new(self.label.clone(), self.shape.clone(), data)
    }
}
