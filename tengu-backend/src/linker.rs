use crate::{Backend, StorageType};

pub trait Linker {
    /// Type of the backend.
    type Backend: Backend;

    /// Type of output.
    type Output;

    fn copy_link<T: StorageType>(
        &mut self,
        from: &<Self::Backend as Backend>::Tensor<T>,
        to: &<Self::Backend as Backend>::Tensor<T>,
    );

    fn finish(self) -> Self::Output;
}
