use crate::Backend;

pub struct Linker;

impl tengu_backend::Linker for Linker {
    type Backend = Backend;

    fn copy_link<T: tengu_backend_tensor::StorageType>(
        &mut self,
        _from: &<Self::Backend as tengu_backend::Backend>::Tensor<T>,
        _to: &<Self::Backend as tengu_backend::Backend>::Tensor<T>,
    ) {
        todo!()
    }
}
