use std::collections::HashSet;
use std::rc::Rc;

use tengu_backend::{Error, Result};
use tengu_backend_tensor::{IOType, StorageType};

use crate::compute::Compute;
use crate::limits::Limits;
use crate::linker::Linker;
use crate::processor::Processor;
use crate::readout::Readout;
use crate::tensor::Tensor;

pub struct Backend;

impl tengu_backend::Backend for Backend {
    type Tensor<T: StorageType> = Tensor<T>;
    type Compute<'a> = Compute;
    type Processor<'a> = Processor<'a>;
    type Linker<'a> = Linker;
    type Readout<'a> = Readout;
    type Limits = Limits;

    async fn new() -> Result<Rc<Self>> {
        Ok(Rc::new(Self))
    }

    fn limits(&self) -> Self::Limits {
        Limits
    }

    fn processor<'a>(&self, readouts: &'a HashSet<String>) -> Self::Processor<'a> {
        Processor::new(readouts)
    }

    fn propagate(&self, call: impl FnOnce(Self::Linker<'_>)) {
        call(Linker);
    }

    fn compute<F>(&self, _label: &str, call: F) -> Result<()>
    where
        F: FnOnce(Self::Compute<'_>) -> anyhow::Result<()>,
    {
        call(Compute).map_err(Error::ComputeError)
    }

    fn readout(&self, _label: &str, call: impl FnOnce(Self::Readout<'_>)) {
        call(Readout);
    }

    fn zero<T: StorageType>(
        self: &Rc<Self>,
        label: impl Into<String>,
        shape: impl Into<Vec<usize>>,
    ) -> Self::Tensor<T> {
        Tensor::empty(label, shape)
    }

    fn tensor<T: IOType>(
        self: &Rc<Self>,
        label: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        data: &[T],
    ) -> Self::Tensor<T> {
        Tensor::new(label, shape, data)
    }
}
