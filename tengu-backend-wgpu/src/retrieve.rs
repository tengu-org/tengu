//! This module defines the `Readout` struct which implements the `Readout` trait from the `tengu_backend` crate.
//! It is used for committing readout operations using an encoder provided by `tengu_wgpu`.

use tengu_backend::{Backend, Result};
use tracing::trace;

use crate::Backend as WGPUBackend;

/// The `Readout` struct is used to perform readout operations in the WGPU backend.
/// It holds a mutable reference to an `Encoder` which is used to encode the readout operations.
pub struct Retrieve;

impl tengu_backend::Retrieve for Retrieve {
    type Backend = WGPUBackend;

    /// Commits the readout operations by iterating over the sources of the processor
    /// and performing the readout operation on each tensor.
    ///
    /// # Parameters
    /// - `processor`: A reference to the processor from the backend which provides the sources.
    async fn run(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>) -> Result<()> {
        trace!("Comitting readout operation...");
        for source in processor.sources() {
            source.retrieve().await?;
        }
        Ok(())
    }
}
