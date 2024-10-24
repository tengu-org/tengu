use tengu_backend::Compute as RawCompute;
use tengu_backend::Result;

use crate::processor::Processor;
use crate::Backend as CPUBackend;

pub struct Compute;

impl RawCompute<CPUBackend> for Compute {
    fn run(&mut self, _processor: &Processor<'_>) -> Result<()> {
        Ok(())
    }
}
