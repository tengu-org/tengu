//! The runner module contains the `Runner` struct, which is responsible for running the graph.
//! The runner executes the graph by computing the blocks, reading out the results, propagating the
//! results through the links, and retrieving the results from the blocks.

use tengu_backend::Backend;

use crate::graph::Block;
use crate::{Error, Result};

use super::link::RealizedLink;
use super::Graph;

/// The `Runner` struct is responsible for running the computational graph.
pub struct Runner<'a, B: Backend> {
    backend: &'a B,
    blocks: Vec<&'a Block<B>>,
    links: Vec<RealizedLink<'a, B>>,
    processors: Vec<B::Processor<'a>>,
}

impl<'a, B: Backend + 'static> Runner<'a, B> {
    /// Creates a new `Runner` instance with the specified graph.
    ///
    /// # Parameters
    /// - `graph`: A reference to the computational graph.
    ///
    /// # Returns
    /// A new `Runner` instance.
    pub fn new(graph: &'a Graph<B>) -> Self {
        let blocks: Vec<_> = graph.blocks.values().collect();
        let links: Vec<_> = graph.links.iter().map(|link| link.realize(graph)).collect();
        let processors = blocks.iter().map(|block| block.processor()).collect();
        Self {
            backend: graph.tengu.backend(),
            blocks,
            links,
            processors,
        }
    }

    /// Executes a single step of computation of the graph.
    ///
    /// # Returns
    /// A result indicating success or failure.
    pub async fn step(&self) -> Result<()> {
        self.compute()?;
        self.readout();
        self.propagate();
        self.retrieve().await
    }

    /// Computes the blocks in the graph.
    ///
    /// # Returns
    /// A result indicating success or failure.
    fn compute(&self) -> Result<()> {
        self.backend
            .compute("compute", |mut compute| {
                for (block, processor) in self.blocks.iter().zip(&self.processors) {
                    block.compute(&mut compute, processor)?;
                }
                Ok(())
            })
            .map_err(Error::BackendError)
    }

    /// Reads out blocks in the graph.
    fn readout(&self) {
        self.backend.readout("readout", |mut readout| {
            for (block, processor) in self.blocks.iter().zip(&self.processors) {
                block.readout(&mut readout, processor);
            }
        });
    }

    /// Retrieves out the results from the blocks in the graph.
    ///
    /// # Returns
    /// A result indicating success or failure.
    async fn retrieve(&self) -> Result<()> {
        self.backend
            .retrieve(|mut retrieve| async move {
                for (block, processor) in self.blocks.iter().zip(&self.processors) {
                    block.retrieve(&mut retrieve, processor).await?;
                }
                Ok(())
            })
            .await
            .map_err(Error::BackendError)
    }

    /// Propagates information through the links in the graph.
    ///
    /// # Returns
    /// A result indicating success or failure.
    fn propagate(&self) {
        self.backend.propagate(|mut linker| {
            for link in &self.links {
                link.from()
                    .copy(link.to(), &mut linker)
                    .expect("link endpoints should match");
            }
        });
    }
}
