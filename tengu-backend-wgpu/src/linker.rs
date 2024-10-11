use tengu_backend::Backend;
use tengu_wgpu::Encoder;

use crate::source::Source;
use crate::Backend as WGPUBackend;

pub struct Linker<'a> {
    encoder: &'a mut Encoder,
}

impl<'a> Linker<'a> {
    pub fn new(encoder: &'a mut Encoder) -> Self {
        Self { encoder }
    }
}

impl<'a> tengu_backend::Linker<'a> for Linker<'a> {
    type Backend = WGPUBackend;

    fn copy_link<T: tengu_backend::StorageType>(
        &mut self,
        from: &<Self::Backend as Backend>::Tensor<T>,
        to: &<Self::Backend as Backend>::Tensor<T>,
    ) {
        self.encoder.copy_buffer(from.buffer(), to.buffer());
    }
}
