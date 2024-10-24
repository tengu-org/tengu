//! This module defines the `Readout` struct which implements the `Readout` trait from the `tengu_backend` crate.
//! It is used for reading out the data from the GPU buffers into staging buffers using the WGPU backend.

use tengu_backend::Readout as RawReadout;
use tengu_wgpu::Encoder;
use tracing::trace;

use crate::processor::Processor;
use crate::Backend as WGPUBackend;

/// The `Readout` struct is used to perform readout operations on the WGPU backend.
pub struct Readout<'a> {
    encoder: &'a mut Encoder,
}

impl<'a> Readout<'a> {
    /// Creates a new `Readout` instance.
    ///
    /// # Parameters
    /// - `encoder`: A mutable reference to an `Encoder` object used for the readout operations.
    ///
    /// # Returns
    /// A new instance of `Readout`.
    pub fn new(encoder: &'a mut Encoder) -> Self {
        Self { encoder }
    }
}

impl<'a> RawReadout<WGPUBackend> for Readout<'a> {
    /// Runs the readout operation by iterating over the sources of the processor
    /// and performing the readout operation on each tensor.
    ///
    /// # Parameters
    /// - `processor`: A reference to the processor from the backend which provides the sources.
    fn run(&mut self, processor: &Processor<'_>) {
        trace!("Executing readout operation");
        for source in processor.readout_sources() {
            source.readout(self.encoder);
        }
    }
}
