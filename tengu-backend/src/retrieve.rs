//! This module defines the `Retrieve` trait, which represents an abstraction for retrieval
//! operations in the Tengu backend. Retrieval operations move the data from staging buffers
//! to the probes that correspond to each specific tensor and its staging buffer.

#![allow(async_fn_in_trait)]

use crate::{Backend, Result};

/// The `Retrieve` trait defines an interface for retrieving data asynchronously
/// from the associated staging buffer.
pub trait Retrieve {
    /// The type of the backend.
    type Backend: Backend;

    /// Runs the retrieve process using the specified processor to provide information about tensors.
    ///
    /// # Parameters
    /// - `processor`: A reference to the processor used for finding tensors and performing
    ///   the retrieval process on them.
    ///
    /// # Returns
    /// A result indicating the success or failure of the readout process.
    async fn run(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>) -> Result<()>;
}
