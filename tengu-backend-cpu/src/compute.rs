//! This module defines the `Compute` struct which implements the `Compute` trait from the `tengu_backend` crate.
//! Since all the work in case of this CPU implementation is done by the processor, `Compute` is
//! essentially a no-op struct.

use tengu_backend::Compute as RawCompute;
use tengu_backend::Result;

use crate::processor::Processor;
use crate::Backend as CPUBackend;

/// The `Compute` struct is used to manage and execute compute passes on the CPU. A new `Compute`
/// struct is create for each execution of the commit pass.
pub struct Compute;

impl RawCompute<CPUBackend> for Compute {
    /// Runs the compute operation. A no-op in this case.
    fn run(&mut self, _processor: &Processor<'_>) -> Result<()> {
        Ok(())
    }
}
