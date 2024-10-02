use tengu_backend::Backend;
use tengu_wgpu::Encoder;

use crate::source::Source;
use crate::Backend as WGPUBackend;

pub struct Linker {
    encoder: Encoder,
}

impl Linker {
    pub fn new(encoder: Encoder) -> Self {
        Self { encoder }
    }
}

impl tengu_backend::Linker for Linker {
    type Backend = WGPUBackend;
    type Output = wgpu::CommandBuffer;

    fn copy_link<T: tengu_backend::StorageType>(
        &mut self,
        from: &<Self::Backend as Backend>::Tensor<T>,
        to: &<Self::Backend as Backend>::Tensor<T>,
    ) {
        self.encoder.copy_buffer(from.buffer(), to.buffer());
    }

    fn finish(self) -> Self::Output {
        self.encoder.finish()
    }
}
