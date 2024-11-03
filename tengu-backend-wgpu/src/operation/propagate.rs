//! This module defines the `Linker` struct which implements the `Linker` trait from the `tengu_backend` crate.
//! The `Linker` is responsible for copying data between GPU buffers using the WGPU backend.

use std::rc::Rc;
use tracing::trace;

use tengu_backend::Pass as RawPass;
use tengu_backend::Propagate as RawPropagate;
use tengu_backend::{Error, Operation, Result};
use tengu_utils::Label;
use tengu_wgpu::{Device, Encoder};

use crate::processor::propagate::Link;
use crate::processor::propagate::Processor;
use crate::Backend as WGPUBackend;

/// The `Linker` struct is used to manage and perform copy operations between GPU buffers.
/// It holds a mutable reference to an `Encoder` which is used to encode the copy operations.
pub struct Propagate {
    device: Rc<Device>,
    label: Label,
}

impl Operation<WGPUBackend> for Propagate {
    type IR<'a> = Vec<Link<'a>>;
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
        let mut encoder = self.device.encoder(self.label.as_ref());
        trace!("Executing propagation step");
        call(Pass::new(&mut encoder)).map_err(Error::PropagateError)?;
        trace!("Submitting propagation commands to the queue");
        self.device.submit(encoder.finish());
        Ok(())
    }
}

impl RawPropagate<WGPUBackend> for Propagate {
    type Processor<'a> = Processor<'a>;

    fn processor(&self) -> Self::Processor<'_> {
        Processor::new()
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
    type IR<'b> = Vec<Link<'b>>;

    fn run(&mut self, ir: &Self::IR<'_>) -> Result<()> {
        for link in ir {
            link.copy(self.encoder);
        }
        Ok(())
    }
}
