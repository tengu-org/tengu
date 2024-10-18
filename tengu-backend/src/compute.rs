//! This module defines the `Compute` trait, which is used for performing computations
//! using a given processor state within the backend. The trait provides an interface
//! for running computation tasks using data accumulated and preprocessed by processor.

use crate::{Backend, Result};

/// The `Compute` trait defines a set of operations for performing computations within a backend.
/// Types that implement this trait can use a processor state to execute computation tasks.
pub trait Compute {
    /// The type of the backend that this compute trait interacts with.
    type Backend: Backend;

    /// Uses the given processor state to perform computations.
    ///
    /// # Parameters
    /// - `processor`: A reference to the processor that was used to prepare data from tensor AST
    ///   for backend consumption.
    ///
    /// # Returns
    /// A `Result` indicating whether the computation was successful or an error occurred.
    fn run(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>) -> Result<()>;
}
