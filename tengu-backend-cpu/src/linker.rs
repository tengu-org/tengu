use tengu_backend::Backend;

use crate::Backend as CPUBackend;

pub struct Linker;

impl tengu_backend::Linker<'_> for Linker {
    type Backend = CPUBackend;

    fn copy_link<T: tengu_backend::StorageType>(
        &mut self,
        from: &<Self::Backend as Backend>::Tensor<T>,
        to: &<Self::Backend as Backend>::Tensor<T>,
    ) {
        to.copy_from(from);
    }
}
