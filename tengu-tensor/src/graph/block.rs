//! Module for defining and managing blocks in the Tengu tensor computation framework.
//!
//! This module defines the `Block` struct and associated methods for creating, managing, and processing
//! computational blocks. Blocks are essential components in the computational graph, responsible for
//! holding and executing computations.
//!
//! A block is an independent unit of computation. It is executed in parallel with all other
//! blocks, and doesn't share tensors with any other blocks. On some backends (like WGPU), a block
//! will have its own memory and its own shader.

use std::rc::Rc;
use tengu_backend::{Backend, Compute, Processor, Readout, StorageType};

use super::computation::Computation;
use crate::expression::{Expression, Shape, Source};
use crate::Tengu;

/// A struct representing a computational block in the Tengu framework.
///
/// The `Block` struct holds computations and provides methods to add, label, and process these computations.
pub struct Block<B: Backend> {
    tengu: Rc<Tengu<B>>,
    label: String,
    computations: Vec<Computation<B>>,
}

impl<B: Backend + 'static> Block<B> {
    /// Creates a new `Block` instance with the specified label.
    ///
    /// # Parameters
    /// - `tengu`: A reference-counted Tengu instance.
    /// - `label`: The label for the new block.
    ///
    /// # Returns
    /// A new `Block` instance.
    pub fn new(tengu: &Rc<Tengu<B>>, label: impl Into<String>) -> Self {
        Self {
            tengu: Rc::clone(tengu),
            label: label.into(),
            computations: Vec::new(),
        }
    }

    /// Returns the label of the block.
    ///
    /// # Returns
    /// A reference to the label string.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Adds a new computation to the block with the specified label and expression.
    ///
    /// # Type Parameters
    /// - `T`: The storage type of the expression.
    ///
    /// # Parameters
    /// - `label`: The label for the new computation.
    /// - `expr`: The expression to be computed.
    ///
    /// # Returns
    /// A mutable reference to the `Block` instance for chaining with other block calls.
    pub fn add_computation<T: StorageType>(&mut self, label: impl Into<String>, expr: Expression<T, B>) -> &mut Self {
        let output = self.tengu.tensor(expr.shape()).label(label).zero::<T>();
        let computation = Computation::new(output, expr);
        self.computations.push(computation);
        self
    }

    /// Executes the computations in the block using the provided compute object and processor.
    ///
    /// # Parameters
    /// - `compute`: A mutable reference to the compute object.
    /// - `processor`: A reference to the processor.
    ///
    /// # Returns
    /// A `Result` indicating whether the computation was successful or an error occurred.
    pub fn compute(&self, compute: &mut B::Compute<'_>, processor: &B::Processor<'_>) -> tengu_backend::Result<()> {
        compute.commit(processor)
    }

    /// Executes the tensor readout operation for all tensors in the block which have a probe
    /// associated with them.
    ///
    /// # Parameters
    /// - `readout`: A mutable reference to the readout object.
    /// - `processor`: A reference to the processor.
    pub fn readout(&self, readout: &mut B::Readout<'_>, processor: &B::Processor<'_>) {
        readout.commit(processor);
    }

    /// Retrieves a source by its label from the computations in the block.
    ///
    /// # Parameters
    /// - `label`: The label of the source to retrieve.
    ///
    /// # Returns
    /// An optional reference to the source.
    pub(crate) fn source(&self, label: &str) -> Option<&dyn Source<B>> {
        self.computations.iter().flat_map(|c| c.source(label)).next()
    }

    /// Creates a processor specific for this block. Adding computations will invalidate the
    /// processor.
    ///
    /// # Returns
    /// A processor for the block.
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
