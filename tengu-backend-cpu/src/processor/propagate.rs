use tengu_backend::Processor as RawProcessor;
use tengu_tensor::{CopyFrom, StorageType};
use tengu_tensor_cpu::Tensor;

use super::Atom;
use crate::Backend as CPUBackend;

pub struct Processor;

impl<'a> RawProcessor<'a, CPUBackend> for Processor {
    type Statement = ();
    type Block = ();

    fn var<T: StorageType>(&mut self, _tensor: &'a Tensor<T>) -> Atom<'a, T> {
        None
    }

    fn scalar<T: StorageType>(&mut self, _value: T) -> Atom<'a, T> {
        None
    }

    fn statement<T: StorageType>(&mut self, _out: &'_ Tensor<T>, _expr: Atom<'_, T>) -> Self::Statement
    where
        Tensor<T>: CopyFrom,
    {
    }

    fn block(&mut self, _exprs: impl IntoIterator<Item = Self::Statement>) -> Self::Block {}

    fn link<T: StorageType>(&mut self, from: &'a Tensor<T>, to: &'a Tensor<T>) {
        to.copy_from(from, &mut ())
    }
}
