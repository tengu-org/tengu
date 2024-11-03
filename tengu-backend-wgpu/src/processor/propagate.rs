use tengu_backend::Processor as RawProcessor;
use tengu_tensor::{CopyFrom, StorageType};
use tengu_tensor_wgpu::Tensor;

use crate::Backend as WGPUBackend;

use super::Atom;
pub use link::Link;

mod link;

pub struct Processor<'a> {
    links: Vec<Link<'a>>,
}

impl<'a> Processor<'a> {
    pub fn new() -> Self {
        Self { links: Vec::new() }
    }
}

impl<'a> RawProcessor<'a, WGPUBackend> for Processor<'a> {
    type Statement = ();
    type Block = ();

    fn var<T: StorageType>(&mut self, _tensor: &'a Tensor<T>) -> Atom<'a, T> {
        None
    }

    fn scalar<T: StorageType>(&mut self, _value: T) -> Atom<'a, T> {
        None
    }

    fn statement<T: StorageType>(&mut self, _out: &'a Tensor<T>, _expr: Atom<'a, T>) -> Self::Statement
    where
        Tensor<T>: CopyFrom,
    {
    }

    fn block(&mut self, _exprs: impl IntoIterator<Item = Self::Statement>) -> Self::Block {}

    fn link<T: StorageType>(&mut self, from: &'a Tensor<T>, to: &'a Tensor<T>) {
        let link = Link::new(from.buffer(), to.buffer());
        self.links.push(link);
    }
}

impl<'a> Default for Processor<'a> {
    fn default() -> Self {
        Self::new()
    }
}
