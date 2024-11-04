//! The runner module contains the `Executor` struct, which is responsible for running the graph.
//! The executor does its jobby computing the blocks, reading out the results, and propagating the
//! results through the links.

use tengu_backend::Backend;

use super::link::RealizedLink;
use super::Graph;
use crate::graph::Block;
use crate::{Error, Result};

/// The `Executor` struct is responsible for running the computational graph.
pub struct Executor<'a, B: Backend> {
    backend: &'a B,
    blocks: Vec<&'a Block<B>>,
    links: Vec<RealizedLink<'a, B>>,
}

impl<'a, B: Backend + 'static> Executor<'a, B> {
    /// Creates a new `Executor` instance with the specified graph.
    ///
    /// # Parameters
    /// - `graph`: A reference to the computational graph.
    ///
    /// # Returns
    /// A new `Executor` instance.
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
    pub fn step(&self) -> Result<()> {
        self.compute()?;
        self.propagate();
        self.readout();
        Ok(())
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

    /// Performs readout operation on blocks in the graph.
    fn readout(&self) {
        self.backend.readout("readout", |mut stage| {
            for (block, processor) in self.blocks.iter().zip(&self.processors) {
                block.readout(&mut stage, processor);
            }
        });
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
