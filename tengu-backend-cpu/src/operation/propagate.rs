//! This module defines the `Propagate` struct which implements the `Linker` trait from the `tengu_backend` crate.
//! The `Propagate` is responsible for copying data between CPU buffers using the CPU backend.

use tengu_backend::Pass as RawPass;
use tengu_backend::Propagate as RawPropagate;
use tengu_backend::{Operation, Result};
use tengu_utils::Label;

use crate::processor::propagate::Processor;
use crate::Backend as CPUBackend;

/// The `Propagate` struct is used to manage and perform copy operations between CPU buffers.
pub struct Propagate;

impl Operation<CPUBackend> for Propagate {
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

impl RawPropagate<CPUBackend> for Propagate {
    type Processor<'a> = Processor;

    fn processor(&self) -> Self::Processor<'_> {
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
