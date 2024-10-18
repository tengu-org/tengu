//! This module defines the `Retrieve` struct which implements the `Retrieve` trait from the `tengu_backend` crate.
//! It is used for running retrieval operations on sources provided by the processor in the WGPU backend.

use tengu_backend::{Backend, Result};
use tracing::trace;

use crate::Backend as WGPUBackend;

/// The `Retrieve` struct is used to perform retrieval operations in the WGPU backend.
pub struct Retrieve;

impl tengu_backend::Retrieve for Retrieve {
    type Backend = WGPUBackend;

    /// Commits the readout operations by iterating over the sources of the processor
    /// and performing the retrieval operation on each tensor.
    ///
    /// # Parameters
    /// - `processor`: A reference to the processor from the backend which provides the sources.
    ///
    /// # Returns
    /// A result indicating the success or failure of the retrieval process.
    async fn run(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>) -> Result<()> {
        trace!("Executing retrieve operation");
        for source in processor.sources() {
            source.retrieve().await?;
        }
        Ok(())
    }
}
