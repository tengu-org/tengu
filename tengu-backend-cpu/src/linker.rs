use tengu_backend::Linker as RawLinker;
use tengu_backend_tensor::StorageType;

use crate::tensor::Tensor;
use crate::Backend as CPUBackend;

pub struct Linker;

impl RawLinker<CPUBackend> for Linker {
    fn copy_link<T: StorageType>(&mut self, from: &Tensor<T>, to: &Tensor<T>) {
        from.copy_to(to);
    }
}
