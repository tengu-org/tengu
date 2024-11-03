//! This module defines the `Readout` struct which implements the `Readout` trait from the `tengu_backend` crate.
//! It is used for reading out the data from the GPU buffers into staging buffers using the WGPU backend.

use std::collections::HashSet;
use std::rc::Rc;

use tengu_backend::Pass as RawPass;
use tengu_backend::Readout as RawReadout;
use tengu_backend::{Error, Operation, Result};
use tengu_utils::Label;
use tengu_wgpu::{Device, Encoder};
use tracing::trace;

use crate::processor::readout;
use crate::processor::readout::Processor;
use crate::Backend as WGPUBackend;

// NOTE: Readout implementation.

/// The `Readout` struct is used to perform readout operations on the WGPU backend.
pub struct Readout {
    device: Rc<Device>,
    label: Label,
}

impl Operation<WGPUBackend> for Readout {
    type IR<'a> = readout::Block<'a>;
    type Pass<'a> = Pass<'a>;

    fn new(backend: &Rc<WGPUBackend>, label: impl Into<Label>) -> Self {
        Self {
            device: Rc::clone(backend.device()),
            label: label.into(),
        }
    }

    fn run<F>(&mut self, call: F) -> Result<()>
    where
        F: FnOnce(Self::Pass<'_>) -> anyhow::Result<()>,
    {
        trace!("Executing readout step");
        let commands = self
            .device
            .encoder(self.label.as_ref())
            .stage(|encoder| call(Pass::new(encoder)))
            .map_err(|e| Error::ReadoutError(e.into()))?
            .finish();
        trace!("Submitting readout commands to the queue");
        self.device.submit(commands);
        Ok(())
    }
}

impl RawReadout<WGPUBackend> for Readout {
    type Processor<'a> = Processor<'a> where Self: 'a;

    fn processor<'a>(&self, readouts: &'a HashSet<Label>) -> Self::Processor<'a> {
        Processor::new(readouts)
    }
}

// NOTE: Pass implementation.

pub struct Pass<'a> {
    encoder: &'a mut Encoder,
}

impl<'a> Pass<'a> {
    pub fn new(encoder: &'a mut Encoder) -> Self {
        Self { encoder }
    }
}

impl<'a> RawPass<WGPUBackend> for Pass<'a> {
    type IR<'b> = readout::Block<'a>;

    fn run(&mut self, ir: &Self::IR<'a>) -> Result<()> {
        trace!("Executing readout operation");
        for source in ir.sources() {
            source.readout(self.encoder);
        }
        Ok(())
    }
}
