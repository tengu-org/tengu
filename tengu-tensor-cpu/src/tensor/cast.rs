use tengu_tensor::{Cast, StorageType};

use super::Tensor;
use crate::primitive_cast::PrimitiveCast;

impl<T, S> Cast<S> for Tensor<T>
where
    T: StorageType + PrimitiveCast<S>,
    S: StorageType,
{
    type Output = Tensor<S>;

    fn cast(&self) -> Self::Output {
        let data: Vec<S> = self.data.borrow().iter().map(|v| (*v).cast()).collect();
        Tensor::from_tensor(self, data)
    }
}
