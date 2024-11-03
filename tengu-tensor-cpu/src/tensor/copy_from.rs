use tengu_tensor::{CopyFrom, StorageType};

use crate::Tensor;

impl<T: StorageType> CopyFrom for Tensor<T> {
    type Linker = ();

    /// Copies data from another tensor into this tensor.
    ///
    /// # Parameters
    /// - `other`: The tensor to copy data from.
    fn copy_from(&self, other: &Self, _liner: &mut Self::Linker) {
        self.data.borrow_mut().copy_from_slice(&other.data.borrow());
    }
}
