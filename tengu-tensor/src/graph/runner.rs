use tengu_backend::Backend;

use crate::graph::Block;
use crate::{Error, Result};

use super::link::RealizedLink;
use super::Graph;

pub struct Runner<'a, B: Backend> {
    backend: &'a B,
    blocks: Vec<&'a Block<B>>,
    links: Vec<RealizedLink<'a, B>>,
    processors: Vec<B::Processor<'a>>,
}

impl<'a, B: Backend + 'static> Runner<'a, B> {
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

    pub async fn step(&self) -> Result<()> {
        self.compute()?;
        self.readout();
        self.propagate();
        self.retrieve().await
    }

    /// Computes the blocks in the graph.
    ///
    /// # Parameters
    /// - `processors`: A vector of processors for the blocks, one processor for each block.
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
    ///
    /// # Parameters
    /// - `processors`: A vector of processors for the blocks, once processor for each block.
    fn readout(&self) {
        self.backend.readout("readout", |mut readout| {
            for (block, processor) in self.blocks.iter().zip(&self.processors) {
                block.readout(&mut readout, processor);
            }
        });
    }

    /// Retrieves out the results from the blocks in the graph.
    ///
    /// # Parameters
    /// - `processors`: A vector of processors for the blocks, once processor for each block.
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

    /// Computes the links in the graph.
    ///
    /// # Parameters
    /// - `links`: A vector of source pairs for the links.
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
