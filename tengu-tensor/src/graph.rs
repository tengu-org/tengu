//! Module for creating and managing computational graphs in the Tengu tensor computation framework.
//!
//! This module defines the `Graph` struct and associated methods for constructing and processing
//! computational graphs. It provides an interface to add blocks, link them, and perform
//! computations using the blocks and links.

use as_any::Downcast;
use futures::Future;
use std::collections::HashMap;
use std::sync::Arc;
use tengu_backend::{Backend, StorageType};

use crate::expression::Source;
use crate::probe::Probe;
use crate::tensor::Tensor;
use crate::{Error, Result, Tengu};

use block::Block;
use link::Link;
use runner::Runner;

mod block;
mod computation;
mod link;
mod runner;

/// A struct representing a computational graph in the Tengu framework.
///
/// The `Graph` struct holds blocks and links, allowing for the construction and processing
/// of complex computations.
pub struct Graph<B: Backend> {
    tengu: Arc<Tengu<B>>,
    blocks: HashMap<String, Block<B>>,
    links: Vec<Link>,
}

// NOTE: Computation interface

impl<B: Backend + 'static> Graph<B> {
    /// Performs computations in the graph for a specified number of iterations.
    ///
    /// # Parameters
    /// - `times`: The number of iterations to perform.
    ///
    /// # Returns
    /// A result indicating success or failure.
    pub async fn compute(&self, times: usize) -> Result<()> {
        let runner = Runner::new(self);
        for _ in 0..times {
            runner.step().await?;
        }
        Ok(())
    }

    /// Processes the graph for a specified number of iterations with a user-defined callback.
    ///
    /// # Parameters
    /// - `times`: The number of iterations to perform.
    /// - `call`: A callback function to call after each iteration.
    ///
    /// # Returns
    /// A result indicating success or failure.
    pub async fn process<F, Fut>(&self, times: usize, mut call: F) -> Result<()>
    where
        Fut: Future,
        F: FnMut(usize) -> Fut,
    {
        let runner = Runner::new(self);
        for i in 0..times {
            runner.step().await?;
            call(i).await;
        }
        Ok(())
    }

    /// Processes the graph for a specified number of iterations with a user-defined callback that
    /// determines whether to continue.
    ///
    /// # Parameters
    /// - `times`: The number of iterations to perform.
    /// - `call`: A callback function that returns a boolean indicating whether to continue processing.
    ///
    /// # Returns
    /// A result indicating success or failure.
    pub async fn process_while<F, Fut>(&self, times: usize, mut call: F) -> Result<()>
    where
        Fut: Future<Output = bool>,
        F: FnMut(usize) -> Fut,
    {
        let runner = Runner::new(self);
        for i in 0..times {
            runner.step().await?;
            if !call(i).await {
                break;
            }
        }
        Ok(())
    }
}

// NOTE: Construction interface

impl<B: Backend + 'static> Graph<B> {
    /// Creates a new `Graph` instance associated with the provided Tengu instance.
    ///
    /// # Parameters
    /// - `tengu`: A reference-counted Tengu instance.
    ///
    /// # Returns
    /// A new `Graph` instance.
    pub fn new(tengu: &Arc<Tengu<B>>) -> Self {
        Self {
            tengu: Arc::clone(tengu),
            blocks: HashMap::new(),
            links: Vec::new(),
        }
    }

    /// Adds a new block to the graph with the specified label.
    ///
    /// # Parameters
    /// - `label`: The label for the new block.
    ///
    /// # Returns
    /// A result containing a mutable reference to the new block or an error if the block already exists.
    pub fn add_block(&mut self, label: impl Into<String>) -> Result<&mut Block<B>> {
        let label = label.into();
        if self.blocks.contains_key(&label) {
            return Err(Error::BlockAlreadyExists(label));
        }
        Ok(self
            .blocks
            .entry(label.clone())
            .or_insert(Block::new(&self.tengu, label)))
    }

    /// Retrieves a reference to a block by its label.
    ///
    /// # Parameters
    /// - `label`: The label of the block to retrieve.
    ///
    /// # Returns
    /// A result containing a reference to the block or an error if the block is not found.
    pub fn get_block(&self, label: &str) -> Result<&Block<B>> {
        let block = self
            .blocks
            .get(label)
            .ok_or_else(|| Error::BlockNotFound(label.to_string()))?;
        Ok(block)
    }

    /// Retrieves a mutable reference to a block by its label.
    ///
    /// # Parameters
    /// - `label`: The label of the block to retrieve.
    ///
    /// # Returns
    /// A result containing a mutable reference to the block or an error if the block is not found.
    pub fn get_block_mut(&mut self, label: &str) -> Result<&mut Block<B>> {
        let block = self
            .blocks
            .get_mut(label)
            .ok_or_else(|| Error::BlockNotFound(label.to_string()))?;
        Ok(block)
    }

    /// Creates a link between two tensors in the graph.
    ///
    /// # Parameters
    /// - `from`: The label of the source tensor, in the format "block/tensor".
    /// - `to`: The label of the destination tensor, in the format "block/tensor".
    ///
    /// # Returns
    /// A result containing a reference to the new link or an error if the link creation fails.
    pub fn link(&mut self, from: impl Into<String>, to: impl Into<String>) -> Result<&Link> {
        let link = Link::new(self, from, to)?;
        self.links.push(link);
        Ok(self.links.last().expect("should have the last link"))
    }

    /// Retrieves a probe for a tensor within a block.
    ///
    /// # Parameters
    /// - `path`: The path to the tensor in the format "block/tensor".
    ///
    /// # Returns
    /// A result containing the probe or an error if the tensor is not found or if there is a type mismatch.
    pub fn get_probe<T: StorageType>(&self, path: &str) -> Result<Probe<T, B>> {
        let source = self
            .get_source(path)?
            .downcast_ref::<Tensor<T, B>>()
            .ok_or_else(|| Error::TypeMismatch)?;
        Ok(source.probe())
    }

    /// Retrieves the source object for a given path.
    ///
    /// # Parameters
    /// - `path`: The path to the source in the format "block/source".
    ///
    /// # Returns
    /// A result containing the source or an error if the block or source is not found.
    fn get_source(&self, path: &str) -> Result<&'_ dyn Source<B>> {
        let (block, source) = path
            .split_once('/')
            .ok_or_else(|| Error::InvalidLinkPath(path.to_string()))?;
        self.blocks
            .get(block)
            .ok_or_else(|| Error::BlockNotFound(block.to_string()))?
            .source(source)
            .ok_or_else(|| Error::SourceNotFound(source.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use crate::Tengu;

    #[tokio::test]
    async fn links() {
        let tengu = Tengu::wgpu().await.unwrap();
        let a = tengu.tensor([1, 2, 3]).label("a").zero::<u32>();
        let b = tengu.tensor([1, 2, 3]).label("b").zero::<u32>();
        let mut graph = tengu.graph();
        graph.add_block("main").unwrap().add_computation("c", a + b);
        let link = graph.link("main/c", "main/a").unwrap();
        assert_eq!(link.from(), "main/c");
        assert_eq!(link.to(), "main/a");
    }

    #[tokio::test]
    #[should_panic]
    async fn link_type_mismatch() {
        let tengu = Tengu::wgpu().await.unwrap();
        let a = tengu.tensor([1, 2, 3]).label("a").zero::<u32>();
        let b = tengu.tensor([1, 2, 3]).label("b").zero::<u32>();
        let mut graph = tengu.graph();
        graph
            .add_block("main")
            .unwrap()
            .add_computation("c", (a + b).cast::<f32>());
        graph.link("main/c", "main/a").unwrap();
    }

    #[tokio::test]
    async fn add_block() {
        let tengu = Tengu::wgpu().await.unwrap();
        let mut graph = tengu.graph();
        graph.add_block("main").unwrap();
        let block = graph.get_block("main").unwrap();
        assert_eq!(block.label(), "main");
    }

    #[tokio::test]
    #[should_panic]
    async fn add_block_again() {
        let tengu = Tengu::wgpu().await.unwrap();
        let mut graph = tengu.graph();
        graph.add_block("main").unwrap();
        graph.add_block("main").unwrap();
    }

    #[tokio::test]
    async fn get_probe() {
        let tengu = Tengu::wgpu().await.unwrap();
        let a = tengu.tensor([1, 2, 3]).label("a").zero::<u32>();
        let b = tengu.tensor([1, 2, 3]).label("b").zero::<u32>();
        let mut graph = tengu.graph();
        graph.add_block("main").unwrap().add_computation("c", a + b);
        graph.get_probe::<u32>("main/c").unwrap();
    }

    #[tokio::test]
    #[should_panic]
    async fn get_probe_type_mismatch() {
        let tengu = Tengu::wgpu().await.unwrap();
        let a = tengu.tensor([1, 2, 3]).label("a").zero::<u32>();
        let b = tengu.tensor([1, 2, 3]).label("b").zero::<u32>();
        let mut graph = tengu.graph();
        graph
            .add_block("main")
            .unwrap()
            .add_computation("c", (a + b).cast::<f32>());
        graph.get_probe::<u32>("main/c").unwrap();
    }
}
