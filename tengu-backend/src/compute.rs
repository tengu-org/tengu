//! This module defines the `Compute` trait, which is used for performing computations
//! using a given processor state within a specified backend. The trait provides an interface
//! for committing computation tasks to the backend.

use crate::{Backend, Result};

/// The `Compute` trait defines a set of operations for performing computations within a specified backend.
/// Types that implement this trait can use a processor state to execute computation tasks.
///
/// # Type Parameters
/// - `Backend`: The type of the backend that this compute trait interacts with.
pub trait Compute {
    /// The type of the backend that this compute trait interacts with.
    type Backend: Backend;

    /// Uses the given processor state to perform computations.
    ///
    /// # Parameters
    /// - `processor`: A reference to the processor state used to perform computations.
    ///
    /// # Returns
    /// A `Result` indicating whether the computation was successful or an error occurred.
    fn run(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>) -> Result<()>;
}
