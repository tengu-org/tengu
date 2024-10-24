//! This module defines the `Readout` struct which implements the `Readout` trait from the `tengu_backend` crate.
//! For this CPU implementation, the whole struct is essentially a no-op, becase there are not
//! staging buffers on the CPU.

use tengu_backend::Readout as RawReadout;

use crate::processor::Processor;
use crate::Backend as CPUBackend;

/// The `Readout` struct is used to perform readout operations on the CPU backend.
pub struct Readout;

impl RawReadout<CPUBackend> for Readout {
    /// Runs the readout operation. A no-op in this case.
    fn run(&mut self, _processor: &Processor<'_>) {}
}
