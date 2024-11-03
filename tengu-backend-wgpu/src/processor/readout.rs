use std::collections::HashSet;

use tengu_backend::Processor as RawProcessor;
use tengu_tensor::{CopyFrom, StorageType};
use tengu_tensor_wgpu::Tensor;
use tengu_utils::Label;

use crate::source::Source;
use crate::Backend as WGPUBackend;

mod block;
pub use block::Block;

use super::Atom;

pub struct Processor<'a> {
    visited: HashSet<&'a str>,
    readouts: &'a HashSet<Label>,
    sources: Vec<&'a dyn Source>,
}

impl<'a> Processor<'a> {
    /// Creates a new `Processor` instance.
    ///
    /// # Parameters
    /// - `readouts`: A reference to a set of readout labels.
    ///
    /// # Returns
    /// A new instance of `Processor`.
    pub fn new(readouts: &'a HashSet<Label>) -> Self {
        Self {
            visited: HashSet::new(),
            readouts,
            sources: Vec::new(),
        }
    }

    /// Returns an iterator over the source tensors acquired from the tensor AST that can be used in
    /// readout operations.
    ///
    /// # Returns
    /// An iterator over source tensor references.
    pub fn sources(&'a self) -> impl Iterator<Item = &'a dyn Source> {
        self.sources.iter().copied()
    }
}

impl<'a> RawProcessor<'a, WGPUBackend> for Processor<'a> {
    type Statement = ();
    type Block = Block<'a>;

    fn var<T: StorageType>(&mut self, tensor: &'a Tensor<T>) -> Atom<'a, T> {
        use tengu_tensor::Tensor;
        let label = Tensor::label(tensor).expect("input tensors should have a label");
        if !self.visited.contains(label) && self.readouts.contains(label) {
            self.sources.push(tensor);
        }
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

    fn block(&mut self, _exprs: impl IntoIterator<Item = Self::Statement>) -> Self::Block {
        Block::new(self.sources.drain(..))
    }

    fn link<T: StorageType>(&mut self, _from: &'a Tensor<T>, _to: &'a Tensor<T>) {}
}
