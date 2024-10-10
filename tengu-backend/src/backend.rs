#![allow(async_fn_in_trait)]

use std::rc::Rc;

use crate::{linker::Linker, readout::Readout, Compute, IOType, Processor, Result, StorageType, Tensor};

pub trait Backend {
    /// The underlying raw tensor type.
    type Tensor<T: StorageType>: Tensor<T>;

    /// The type of the node processor that will construction computation object.
    type Processor<'a>: Processor<'a, Backend = Self>
    where
        Self: 'a;

    /// The underlying raw compute type.
    type Compute<'a>: Compute<Backend = Self>
    where
        Self: 'a;

    /// The underlying linker type.
    type Linker<'a>: Linker<'a, Backend = Self>;

    /// The underlying readout type.
    type Readout<'a>: Readout<'a, Backend = Self>;

    /// Create a new backend instance.
    async fn new() -> Result<Rc<Self>>;

    /// Create interpreter to perform recursive computation.
    fn processor(&self) -> Self::Processor<'_>;

    /// Propagate buffers through links.
    fn propagate(&self, call: impl FnOnce(Self::Linker<'_>));

    /// Read out probes.
    fn readout(&self, label: &str, call: impl FnOnce(Self::Readout<'_>));

    /// Compute the specified function on the backend.
    fn compute(&self, label: &str, call: impl FnOnce(Self::Compute<'_>));

    /// Create a new zero-initialized tensor with the specified label and size.
    fn zero<T: StorageType>(self: &Rc<Self>, label: impl Into<String>, count: usize) -> Self::Tensor<T>;

    /// Create a new tensor with the specified label and data.
    fn tensor<T: IOType>(self: &Rc<Self>, label: impl Into<String>, data: &[T]) -> Self::Tensor<T>;

    /// Create a new probe with the specified label and size.
    fn probe<T: StorageType>(self: &Rc<Self>, count: usize) -> <Self::Tensor<T> as Tensor<T>>::Probe;
}
