use tengu_backend_tensor::UnaryFn;

use super::Tensor;

impl UnaryFn for Tensor<f32> {
    fn exp(&self) -> Self {
        let data = self.data.borrow().iter().map(|v| v.exp()).collect::<Vec<_>>();
        Self::new("", self.shape.clone(), data)
    }

    fn log(&self) -> Self {
        let data = self.data.borrow().iter().map(|v| v.ln()).collect::<Vec<_>>();
        Self::new("", self.shape.clone(), data)
    }
}
