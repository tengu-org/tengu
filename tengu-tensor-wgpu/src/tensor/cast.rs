use tengu_tensor::{Cast, StorageType};

use crate::rep::Rep;

use super::Tensor;

impl<T, S> Cast<S> for Tensor<T>
where
    T: StorageType,
    S: StorageType + Rep,
{
    type Output = Tensor<S>;

    fn cast(&self) -> Self::Output {
        Tensor::like(self, format!("{}({})", S::as_str(), self.expression))
    }
}
