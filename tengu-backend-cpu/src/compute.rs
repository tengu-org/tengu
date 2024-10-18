use tengu_backend::{Backend, Error, Result};

use crate::processor::Processor;
use crate::Backend as CPUBackend;

const WORKGROUP_SIZE: u32 = 64;

pub struct Compute;

impl<'a> tengu_backend::Compute for Compute<'a> {
    type Backend = WGPUBackend;

    fn commit(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>) -> Result<()> {
        Ok(processor.compute())
    }
}
