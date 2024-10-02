use tengu_backend::Backend;
use tengu_wgpu::Encoder;

use crate::Backend as WGPUBackend;

pub struct Readout<'a> {
    encoder: &'a mut Encoder,
}

impl<'a> Readout<'a> {
    pub fn new(encoder: &'a mut Encoder) -> Self {
        Self { encoder }
    }
}

impl<'a> tengu_backend::Readout<'a> for Readout<'a> {
    type Backend = WGPUBackend;

    fn commit(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>) {
        processor.sources().for_each(|tensor| tensor.readout(self.encoder));
    }
}
