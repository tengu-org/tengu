//! This module defines the `Readout` struct which implements the `Readout` trait from the `tengu_backend` crate.
//! For this CPU implementation, the whole struct is essentially a no-op, becase there are not
//! staging buffers on the CPU.

use std::collections::HashSet;

use tengu_backend::Pass as RawPass;
use tengu_backend::Readout as RawReadout;
use tengu_backend::{Operation, Result};
use tengu_utils::Label;

use crate::processor::readout::Processor;
use crate::Backend as CPUBackend;

// NOTE Compute implementation.

/// The `Readout` struct is used to perform readout operations on the CPU backend.
pub struct Readout;

impl Operation<CPUBackend> for Readout {
    type IR<'a> = ();
    type Pass<'a> = Pass;

    fn new(_backend: &std::rc::Rc<CPUBackend>, _label: impl Into<Label>) -> Self {
        Self
    }

    /// Runs the readout operation. A no-op in this case.
    fn run<F>(&mut self, _call: F) -> Result<()>
    where
        F: FnOnce(Self::Pass<'_>) -> anyhow::Result<()>,
    {
        Ok(())
    }
}

impl RawReadout<CPUBackend> for Readout {
    type Processor<'a> = Processor;

    fn processor(&self, _readouts: &HashSet<Label>) -> Self::Processor<'_> {
        Processor
    }
}

// NOTE: Pass implementation.

pub struct Pass;

impl RawPass<CPUBackend> for Pass {
    type IR<'a> = ();

    fn run(&mut self, _ir: &Self::IR<'_>) -> Result<()> {
        Ok(())
    }
}
