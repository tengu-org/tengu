//! This module defines the `Readout` struct which implements the `Readout` trait from the `tengu_backend` crate.
//! It is used for committing readout operations using an encoder provided by `tengu_wgpu`.

use tengu_backend::Backend;
use tengu_wgpu::Encoder;

use crate::Backend as WGPUBackend;

/// The `Readout` struct is used to perform readout operations in the WGPU backend.
/// It holds a mutable reference to an `Encoder` which is used to encode the readout operations.
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

impl<'a> tengu_backend::Readout<'a> for Readout<'a> {
    type Backend = WGPUBackend;

    /// Commits the readout operations by iterating over the sources of the processor
    /// and performing the readout operation on each tensor.
    ///
    /// # Parameters
    /// - `processor`: A reference to the processor from the backend which provides the sources.
    fn commit(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>) {
        processor.sources().for_each(|tensor| tensor.readout(self.encoder));
    }
}
