//! This module defines the `Readout` trait, which represents an abstraction for readout
//! operations in the Tengu backend. Readout operations are essential for extracting results
//! from the tensor graph after computations have been completed. The desitnation of a readout
//! operation is a staging buffer which stores the results on the backend side until it is
//! sent to the respective probe.

use crate::Backend;

/// A trait for handling readout operations in the Tengu backend.
///
/// This trait defines the necessary operations and associated types required for
/// extracting results from computations performed on the backend. Implementors of
/// this trait must specify the type of backend used and provide a method for running
/// the readout process.
pub trait Readout<B: Backend> {
    /// Runs the readout process using the specified processor to provider information about tensors.
    ///
    /// # Parameters
    /// - `processor`: A reference to the processor used for finding tensors and performing
    ///   the staging operation on them.
    fn run(&mut self, processor: &B::Processor<'_>);
}
