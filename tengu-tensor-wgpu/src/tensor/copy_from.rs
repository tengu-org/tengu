use tengu_tensor::{CopyFrom, StorageType};
use tengu_wgpu::Encoder;

use crate::Tensor;

impl<T: StorageType> CopyFrom for Tensor<T> {
    type Linker = Encoder;

    /// Copies data from another tensor into this tensor.
    ///
    /// # Parameters
    /// - `other`: The tensor to copy data from.
    fn copy_from(&self, other: &Self, linker: &mut Self::Linker) {
        linker.copy_buffer(other.buffer(), self.buffer());
    }
}
