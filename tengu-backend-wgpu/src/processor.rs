use std::collections::{BTreeMap, HashSet};
use tengu_backend::{Backend, StorageType};

use crate::{source::Source, Backend as WGPUBackend};
use declarator::Declarator;
use emitter::Emitter;

mod declarator;
mod emitter;

pub struct Processor<'a> {
    emitter: Emitter,
    declarator: Declarator<'a>,
    element_count: usize,
    shader: String,
    bound_sources: HashSet<&'a str>,
    sources: BTreeMap<usize, &'a dyn Source>,
    current_binding: usize,
}

impl<'a> Processor<'a> {
    pub fn new() -> Self {
        Self {
            emitter: Emitter::new(),
            declarator: Declarator::new(),
            element_count: 0,
            shader: String::new(),
            bound_sources: HashSet::new(),
            sources: BTreeMap::new(),
            current_binding: 0,
        }
    }

    pub fn element_count(&self) -> usize {
        self.element_count
    }

    pub fn sources(&'a self) -> impl Iterator<Item = &'a dyn Source> {
        self.sources.values().copied()
    }

    pub fn shader(&self) -> &str {
        &self.shader
    }
}

// Processor trait implementation

impl<'a> tengu_backend::Processor<'a> for Processor<'a> {
    type Backend = WGPUBackend;
    type Repr = (usize, String);

    fn var<T: StorageType>(&mut self, tensor: &'a <Self::Backend as Backend>::Tensor<T>) -> Self::Repr {
        if !self.bound_sources.contains(tensor.label()) {
            self.declarator.var(self.current_binding, tensor);
            self.sources.insert(self.current_binding, tensor);
            self.bound_sources.insert(tensor.label());
            self.current_binding += 1;
        }
        (tensor.count(), self.emitter.var(tensor))
    }

    fn scalar<T: StorageType>(&mut self, value: T) -> Self::Repr {
        self.element_count = 0;
        (0, self.emitter.scalar(value))
    }

    fn unary_fn(&mut self, inner: Self::Repr, symbol: &str) -> Self::Repr {
        let expression = self.emitter.unary_fn(inner.1, symbol);
        let element_count = inner.0;
        (element_count, expression)
    }

    fn binary(&mut self, lhs: Self::Repr, rhs: Self::Repr, symbol: &str) -> Self::Repr {
        let expression = self.emitter.binary(lhs.1, rhs.1, symbol);
        let element_count = lhs.0.max(rhs.0);
        (element_count, expression)
    }

    fn cast(&mut self, inner: Self::Repr, ty: &str) -> Self::Repr {
        let expression = self.emitter.cast(inner.1, ty);
        let element_count = inner.0;
        (element_count, expression)
    }

    fn statement(&mut self, out: Self::Repr, expr: Self::Repr) -> Self::Repr {
        let expression = self.emitter.statement(out.1, expr.1);
        let element_count = out.0.max(expr.0);
        (element_count, expression)
    }

    fn block(&mut self, exprs: impl Iterator<Item = Self::Repr>) {
        let (count_exprs, emit_exprs): (Vec<_>, Vec<_>) = exprs.unzip();
        self.emitter.block(emit_exprs.into_iter());
        self.element_count = count_exprs
            .into_iter()
            .max()
            .expect("block should have at least one computation");
        let header = self.declarator.header();
        let body = self.emitter.body();
        self.shader = format!("{}\n\n{}", header, body);
    }
}

// Default implementation

impl<'a> Default for Processor<'a> {
    fn default() -> Self {
        Processor::new()
    }
}
