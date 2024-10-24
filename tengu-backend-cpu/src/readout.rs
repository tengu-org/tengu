use tengu_backend::Readout as RawReadout;

use crate::processor::Processor;
use crate::Backend as CPUBackend;

pub struct Readout;

impl RawReadout<CPUBackend> for Readout {
    fn run(&mut self, _processor: &Processor<'_>) {}
}
