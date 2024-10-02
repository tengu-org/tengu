use crate::{source::Source, Backend as WGPUBackend};
use tengu_backend::{Backend, StorageType};

use collector::CollectingProcessor;
use shader_emit::ShaderEmitProcessor;

mod collector;
mod shader_emit;

pub struct Processor<'a> {
    shader_emit: ShaderEmitProcessor,
    collector: CollectingProcessor<'a>,
    shader: String,
}

impl<'a> Processor<'a> {
    pub fn new() -> Self {
        Self {
            shader_emit: ShaderEmitProcessor::new(),
            collector: CollectingProcessor::new(),
            shader: String::new(),
        }
    }

    pub fn element_count(&self) -> usize {
        self.collector.count()
    }

    pub fn get_source(&self, label: &str) -> Option<&'a dyn Source> {
        self.collector.get_source(label)
    }

    pub fn sources(&'a self) -> impl Iterator<Item = &'a dyn Source> {
        self.collector.sources()
    }

    pub fn shader(&self) -> &str {
        &self.shader
    }
}

// Processor trait implementation

impl<'a> tengu_backend::Processor<'a> for Processor<'a> {
    type Backend = WGPUBackend;
    type Repr = String;

    fn var<T: StorageType>(&mut self, tensor: &'a <Self::Backend as Backend>::Tensor<T>) -> Self::Repr {
        self.collector.var(tensor);
        self.shader_emit.var(tensor)
    }

    fn scalar<T: StorageType>(&mut self, value: T) -> Self::Repr {
        self.shader_emit.scalar(value)
    }

    fn unary_fn(&mut self, inner: Self::Repr, symbol: &str) -> Self::Repr {
        self.shader_emit.unary_fn(inner, symbol)
    }

    fn binary(&mut self, lhs: Self::Repr, rhs: Self::Repr, symbol: &str) -> Self::Repr {
        self.shader_emit.binary(lhs, rhs, symbol)
    }

    fn cast(&mut self, inner: Self::Repr, ty: &str) -> Self::Repr {
        self.shader_emit.cast(inner, ty)
    }

    fn statement(&mut self, out: Self::Repr, expr: Self::Repr) -> Self::Repr {
        self.shader_emit.statement(out, expr)
    }

    fn block(&mut self, exprs: impl Iterator<Item = Self::Repr>) {
        self.shader_emit.block(exprs);
        self.shader = self.shader_emit.shader()
    }
}

// Default implementation

impl<'a> Default for Processor<'a> {
    fn default() -> Self {
        Processor::new()
    }
}
