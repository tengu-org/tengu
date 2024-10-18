//! Readout abstraction for the Tengu backend.
//!
//! This module defines the `Readout` trait, which represents an abstraction for readout
//! operations in the Tengu backend. Readout operations are essential for extracting results
//! from the tensor graph after computations have been completed.
//!
//! # Traits
//!
//! - `Readout`: A trait for handling readout operations in the Tengu backend.
//!
//! # Related Traits
//!
//! - `Backend`: A trait representing the backend where computations are performed.

#![allow(async_fn_in_trait)]

use crate::{Backend, Result};

/// A trait for handling readout operations in the Tengu backend.
///
/// This trait defines the necessary operations and associated types required for
/// extracting results from computations performed on the backend. Implementors of
/// this trait must specify the type of backend used and provide a method for committing
/// the readout process.
///
/// # Associated Types
/// - `Backend`: The type of the backend used for computations.
///
/// # Methods
/// - `commit`: Runs the readout process using the specified processor.
pub trait Readout<'a> {
    /// The type of the backend.
    type Backend: Backend;

    /// Runs the readout process using the specified processor to locate tensors.
    ///
    /// # Parameters
    /// - `processor`: A reference to the processor used for finding tensors and performing
    ///   the readout process on them.
    ///
    /// # Returns
    /// A result indicating the success or failure of the readout process.
    async fn commit(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>) -> Result<()>;
}
