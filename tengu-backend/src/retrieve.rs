#![allow(async_fn_in_trait)]

use crate::{Backend, Result};

pub trait Retrieve {
    /// The type of the backend.
    type Backend: Backend;

    /// Runs the retrieve process using the specified processor to locate tensors.
    ///
    /// # Parameters
    /// - `processor`: A reference to the processor used for finding tensors and performing
    ///   the retrieval process on them.
    ///
    /// # Returns
    /// A result indicating the success or failure of the readout process.
    async fn run(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>) -> Result<()>;
}
