use crate::Backend;

pub struct Linker;

impl tengu_backend::Linker for Linker {
    type Backend = Backend;

    fn copy_link<T: tengu_backend_tensor::StorageType>(
        &mut self,
        from: &<Self::Backend as tengu_backend::Backend>::Tensor<T>,
        to: &<Self::Backend as tengu_backend::Backend>::Tensor<T>,
    ) {
        from.copy_to(to);
    }
}
