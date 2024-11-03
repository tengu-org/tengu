//! This module defines the `Linker` trait, which is used for propagating tensor data between different parts
//! of a computation graph. The trait provides an interface for copying tensor data on a specified backend.

use crate::{Backend, Operation, Processor};

/// The `Linker` trait defines a set of operations for propagating tensor data within a computation graph.
/// Types that implement this trait can manage the copying of tensor data between different storage
/// locations or parts of the computation graph.
pub trait Propagate<B: Backend>: Operation<B> {
    type Processor<'a>: Processor<'a, B>
    where
        Self: 'a;

    fn processor(&self) -> Self::Processor<'_>;
}
