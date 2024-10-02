use std::collections::HashMap;
use tengu_backend::{Backend, StorageType};

use crate::source::Source;
use crate::Backend as WGPUBackend;

pub struct CollectingProcessor<'a> {
    sources: HashMap<&'a str, &'a dyn Source>,
    count: usize,
}

impl<'a> CollectingProcessor<'a> {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            count: 0,
        }
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn get_source(&self, label: &str) -> Option<&'a dyn Source> {
        self.sources.get(label).copied()
    }

    pub fn sources(&'a self) -> impl Iterator<Item = &'a dyn Source> {
        self.sources.values().copied()
    }
}

impl<'a> tengu_backend::Processor<'a> for CollectingProcessor<'a> {
    type Backend = WGPUBackend;
    type Repr = ();

    fn var<T: StorageType>(&mut self, tensor: &'a <Self::Backend as Backend>::Tensor<T>) -> Self::Repr {
        self.sources.insert(tensor.label(), tensor);
        self.count = self.count.max(tensor.count());
    }

    fn scalar<T: StorageType>(&mut self, _value: T) -> Self::Repr {}

    fn unary_fn(&mut self, _inner: Self::Repr, _symbol: &str) -> Self::Repr {}

    fn binary(&mut self, _lhs: Self::Repr, _rhs: Self::Repr, _symbol: &str) -> Self::Repr {}

    fn cast(&mut self, _inner: Self::Repr, _ty: &str) -> Self::Repr {}

    fn statement(&mut self, _out: Self::Repr, _expr: Self::Repr) -> Self::Repr {}

    fn block(&mut self, _exprs: impl Iterator<Item = Self::Repr>) {}
}
