//! This module defines the `Compute` trait, which is used for performing computations
//! using a given processor state within the backend. The trait provides an interface
//! for running computation tasks using data accumulated and preprocessed by processor.

use crate::operation::Operation;
use crate::{Backend, Processor};

/// The `Compute` trait defines a set of operations for performing computations within a backend.
/// Types that implement this trait can use a processor state to execute computation tasks.
pub trait Compute<B: Backend>: Operation<B> {
    type Processor<'a>: Processor<'a, B>
    where
        Self: 'a;

    fn processor(&self) -> Self::Processor<'_>;
}
