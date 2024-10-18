use std::collections::HashMap;
use std::marker::PhantomData;
use tengu_backend::{Backend, StorageType};

use crate::source::Source;
use crate::Backend as CPUBackend;

// NOTE: Expression implementation

pub trait Computation {}

pub struct Expression<'a, T: StorageType> {
    function: Box<dyn (Fn() -> &'a dyn Computation) + 'a>,
    phantom: PhantomData<T>,
}

// NOTE: Processor implementation

pub struct Processor<'a> {
    sources: HashMap<&'a str, &'a dyn Source>,
    function: Option<Box<dyn (Fn() -> &'a dyn Source) + 'a>>,
}

impl<'a> Processor<'a> {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
            function: None,
        }
    }

    /// Returns an iterator over the source tensors acquird from the tensor AST.
    ///
    /// # Returns
    /// An iterator over source tensor references.
    pub fn sources(&'a self) -> impl Iterator<Item = &'a dyn Source> {
        self.sources.values().copied()
    }

    pub fn compute(&'a self) -> &'a dyn Source {
        self.function
            .as_ref()
            .expect("Processor should have a function before executing")()
    }
}

// NOTE: Processor trait implementation

impl<'a> tengu_backend::Processor<'a> for Processor<'a> {
    type Backend = CPUBackend;
    type Repr = Box<dyn (Fn() -> &'a dyn Source) + 'a>;

    fn var<T: StorageType>(&mut self, tensor: &'a <Self::Backend as Backend>::Tensor<T>) -> Self::Repr {
        self.sources.insert(tensor.label(), tensor);
        Box::new(|| tensor)
    }

    fn scalar<T: StorageType>(&mut self, _value: T) -> Self::Repr {
        todo!()
    }

    fn unary_fn(&mut self, _inner: Self::Repr, _symbol: &str) -> Self::Repr {
        todo!()
    }

    fn binary(&mut self, _lhs: Self::Repr, _rhs: Self::Repr, _symbol: &str) -> Self::Repr {
        todo!()
    }

    fn cast(&mut self, _inner: Self::Repr, _ty: &str) -> Self::Repr {
        todo!()
    }

    fn statement(&mut self, _out: Self::Repr, _expr: Self::Repr) -> Self::Repr {
        todo!()
    }

    fn block(&mut self, _exprs: impl Iterator<Item = Self::Repr>) {
        todo!()
    }
}

// Default implementation

impl<'a> Default for Processor<'a> {
    fn default() -> Self {
        Processor::new()
    }
}
