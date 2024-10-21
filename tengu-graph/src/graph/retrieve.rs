//! This module defines the `Retriever` struct and associated functionality for sending tensor data to their respective probes.

use tengu_backend::Backend;

use super::Graph;
use crate::collector::Collector;
use crate::graph::Block;
use crate::Result;

/// The `Retriever` struct is responsible for sending tensor data to their respective probes.
pub struct Retriever<'a, B: Backend> {
    blocks: Vec<&'a Block<B>>,
    collectors: Vec<Collector<'a, B>>,
}

impl<'a, B: Backend + 'static> Retriever<'a, B> {
    /// Creates a new `Retriever` instance with the specified graph.
    ///
    /// # Parameters
    /// - `graph`: A reference to the computational graph.
    ///
    /// # Returns
    /// A new `Retriever` instance.
    pub fn new(graph: &'a Graph<B>) -> Self {
        let blocks: Vec<_> = graph.blocks.values().collect();
        let collectors = blocks.iter().map(|block| block.collector()).collect();
        Self { blocks, collectors }
    }

    /// Retrieves data from tesnors in the graph into the associated probes.
    ///
    /// # Returns
    /// A result indicating success or failure.
    pub async fn step(&self) -> Result<()> {
        for (block, collector) in self.blocks.iter().zip(&self.collectors) {
            block.retrieve(collector).await?;
        }
        Ok(())
    }
}
