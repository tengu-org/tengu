use std::rc::Rc;
use tengu_backend::{Backend, Compute, Processor, Readout, StorageType};

use super::computation::Computation;
use crate::expression::{Expression, Shape, Source};
use crate::Tengu;

// Block implementation

pub struct Block<B: Backend> {
    tengu: Rc<Tengu<B>>,
    label: String,
    computations: Vec<Computation<B>>,
}

impl<B: Backend + 'static> Block<B> {
    pub fn new(tengu: &Rc<Tengu<B>>, label: impl Into<String>) -> Self {
        Self {
            tengu: Rc::clone(tengu),
            label: label.into(),
            computations: Vec::new(),
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn add_computation<T: StorageType>(&mut self, label: impl Into<String>, expr: Expression<T, B>) -> &mut Self {
        let output = self.tengu.tensor(expr.shape()).label(label).zero::<T>();
        let computation = Computation::new(output, expr);
        self.computations.push(computation);
        self
    }

    pub fn compute(&self, compute: &mut B::Compute<'_>, processor: &B::Processor<'_>) {
        compute.commit(processor);
    }

    pub fn readout(&self, readout: &mut B::Readout<'_>, processor: &B::Processor<'_>) {
        readout.commit(processor);
    }

    pub fn source(&self, label: &str) -> Option<&dyn Source<B>> {
        self.computations.iter().flat_map(|c| c.source(label)).next()
    }

    pub fn processor(&self) -> B::Processor<'_> {
        let mut processor = self.tengu.backend().processor();
        let mut statements = Vec::new();
        for computation in &self.computations {
            statements.push(computation.visit(&mut processor));
        }
        processor.block(statements.into_iter());
        processor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn add_computation() {
        let tengu = Tengu::wgpu().await.unwrap();
        let mut graph = tengu.graph();
        let block = graph.add_block("main").unwrap();
        assert_eq!(block.computations.len(), 0);
        block.add_computation("one", tengu.scalar(1));
        assert_eq!(block.computations.len(), 1);
    }
}
