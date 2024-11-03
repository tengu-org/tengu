//! This module defines the `Readout` trait, which represents an abstraction for readout
//! operations in the Tengu backend. Readout operations are essential for extracting results
//! from the tensor graph after computations have been completed. The desitnation of a readout
//! operation is a staging buffer which stores the results on the backend side until it is
//! sent to the respective probe.

use std::collections::HashSet;

use tengu_utils::Label;

use crate::operation::Operation;
use crate::{Backend, Processor};

/// A trait for handling readout operations in the Tengu backend.
///
/// This trait defines the necessary operations and associated types required for
/// extracting results from computations performed on the backend. Implementors of
/// this trait must specify the type of backend used and provide a method for running
/// the readout process.
pub trait Readout<B: Backend>: Operation<B> {
    type Processor<'a>: Processor<'a, B>
    where
        Self: 'a;

    fn processor<'a>(&self, readouts: &'a HashSet<Label>) -> Self::Processor<'a>;
}
