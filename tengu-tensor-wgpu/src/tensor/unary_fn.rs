use tengu_tensor::{StorageType, UnaryFn};

use super::Tensor;

impl<T: StorageType> UnaryFn for Tensor<T> {
    fn exp(&self) -> Self {
        Tensor::like(self, format!("exp({})", self.expression))
    }

    fn log(&self) -> Self {
        Tensor::like(self, format!("log({})", self.expression))
    }
}
