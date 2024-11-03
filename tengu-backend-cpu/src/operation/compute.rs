//! This module defines the `Compute` struct which implements the `Compute` trait from the `tengu_backend` crate.
//! Since all the work in case of this CPU implementation is done by the processor, `Compute` is
//! essentially a no-op struct.

use tengu_backend::Compute as RawCompute;
use tengu_backend::Pass as RawPass;
use tengu_backend::{Operation, Result};
use tengu_utils::Label;

use crate::processor::compute::Processor;
use crate::Backend as CPUBackend;

// NOTE: Compute implementation.

/// The `Compute` struct is used to manage and execute compute passes on the CPU. A new `Compute`
/// struct is create for each execution of the commit pass.
pub struct Compute;

impl Operation<CPUBackend> for Compute {
    type IR<'a> = ();
    type Pass<'a> = Pass;

    fn new(_backend: &std::rc::Rc<CPUBackend>, _label: impl Into<Label>) -> Self {
        Self
    }

    fn run<F>(&mut self, _call: F) -> Result<()>
    where
        F: FnOnce(Self::Pass<'_>) -> anyhow::Result<()>,
    {
        Ok(())
    }
}

impl RawCompute<CPUBackend> for Compute {
    type Processor<'a> = Processor;

    fn processor(&self) -> Self::Processor<'_> {
        Processor
    }
}

// NOTE: Pass implementation.

pub struct Pass;

impl RawPass<CPUBackend> for Pass {
    type IR<'a> = ();

    /// Runs the compute operation. A no-op in this case.
    fn run(&mut self, _ir: &Self::IR<'_>) -> Result<()> {
        Ok(())
    }
}
