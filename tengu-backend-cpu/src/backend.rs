use std::rc::Rc;

use tengu_backend_tensor::StorageType;

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

    async fn new() -> tengu_backend::Result<std::rc::Rc<Self>> {
        todo!()
    }

    fn limits(&self) -> Self::Limits {
        Limits
    }

    fn processor<'a>(&self, readouts: &'a std::collections::HashSet<String>) -> Self::Processor<'a> {
        Processor::new(readouts)
    }

    fn propagate(&self, _call: impl FnOnce(Self::Linker<'_>)) {
        todo!();
    }

    fn compute<F>(&self, _label: &str, _call: F) -> tengu_backend::Result<()>
    where
        F: FnOnce(Self::Compute<'_>) -> anyhow::Result<()>,
    {
        todo!();
    }

    fn readout(&self, _label: &str, _call: impl FnOnce(Self::Readout<'_>)) {
        todo!();
    }

    fn tensor<T: tengu_backend_tensor::IOType>(
        self: &Rc<Self>,
        label: impl Into<String>,
        shape: impl Into<Vec<usize>>,
        data: &[T],
    ) -> Self::Tensor<T> {
        Tensor::new(label, shape, data)
    }

    fn zero<T: StorageType>(
        self: &Rc<Self>,
        label: impl Into<String>,
        shape: impl Into<Vec<usize>>,
    ) -> Self::Tensor<T> {
        Tensor::empty(label, shape)
    }
}
