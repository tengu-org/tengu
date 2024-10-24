//! Module for creating and managing computational graphs in the Tengu tensor computation framework.
//!
//! This module defines the `Graph` struct and associated methods for constructing and processing
//! computational graphs. It provides an interface to add blocks, link them, and perform
//! computations using the blocks and links.

use std::collections::HashMap;
use std::rc::Rc;

use as_any::Downcast;
use futures::Future;
use tengu_backend::Backend;
use tengu_backend_tensor::StorageType;
use tengu_graph_tensor::{Probe, Tensor};

use crate::source::Source;
use crate::{Error, Result, Tengu};

use block::Block;
use executor::Executor;
use link::Link;
use retrieve::Retriever;

mod block;
mod computation;
mod executor;
mod link;
mod retrieve;

/// A struct representing a computational graph in the Tengu framework.
///
/// The `Graph` struct holds blocks and links, allowing for the construction and processing
/// of complex computations.
pub struct Graph<B: Backend> {
    tengu: Rc<Tengu<B>>,
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
        let executor = Executor::new(self);
        let readout = Retriever::new(self);
        for _ in 0..times {
            executor.step()?;
            readout.step().await?;
        }
        Ok(())
    }

    /// Processes the graph for a specified number of iterations with a user-defined async callback.
    ///
    /// # Parameters
    /// - `times`: The number of iterations to perform.
    /// - `call`: A callback function to call after each iteration.
    ///
    /// # Returns
    /// A result indicating success or failure.
    pub async fn process_async<F, Fut>(&self, times: usize, mut call: F) -> Result<()>
    where
        Fut: Future,
        F: FnMut(usize) -> Fut,
    {
        let executor = Executor::new(self);
        let readout = Retriever::new(self);
        for i in 0..times {
            executor.step()?;
            readout.step().await?;
            call(i).await;
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
    pub async fn process(&self, times: usize, mut call: impl FnMut(usize)) -> Result<()> {
        let executor = Executor::new(self);
        let readout = Retriever::new(self);
        for i in 0..times {
            executor.step()?;
            readout.step().await?;
            call(i);
        }
        Ok(())
    }

    /// Processes the graph for a specified number of iterations with a user-defined async callback that
    /// determines whether to continue.
    ///
    /// # Parameters
    /// - `times`: The number of iterations to perform.
    /// - `call`: A callback function that returns a boolean indicating whether to continue processing.
    ///
    /// # Returns
    /// A result indicating success or failure.
    pub async fn process_while_async<F, Fut>(&self, times: usize, mut call: F) -> Result<()>
    where
        Fut: Future<Output = bool>,
        F: FnMut(usize) -> Fut,
    {
        let executor = Executor::new(self);
        let readout = Retriever::new(self);
        for i in 0..times {
            executor.step()?;
            readout.step().await?;
            if !call(i).await {
                break;
            }
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
    pub async fn process_while<F>(&self, times: usize, mut call: F) -> Result<()>
    where
        F: FnMut(usize) -> bool,
    {
        let executor = Executor::new(self);
        let readout = Retriever::new(self);
        for i in 0..times {
            executor.step()?;
            readout.step().await?;
            if !call(i) {
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
    pub fn new(tengu: &Rc<Tengu<B>>) -> Self {
        Self {
            tengu: Rc::clone(tengu),
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
    pub fn add_link(&mut self, from: impl Into<String>, to: impl Into<String>) -> Result<&Link> {
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
    pub fn add_probe<T: StorageType>(&mut self, path: &str) -> Result<Probe<T>> {
        let (block_label, source_label) = path
            .split_once('/')
            .ok_or_else(|| Error::InvalidLinkPath(path.to_string()))?;
        let block = self
            .blocks
            .get_mut(block_label)
            .ok_or_else(|| Error::BlockNotFound(block_label.to_string()))?;
        let source = block
            .source(source_label)
            .ok_or_else(|| Error::SourceNotFound(source_label.to_string()))?
            .downcast_ref::<Tensor<T, B>>()
            .ok_or_else(|| Error::TypeMismatch)?;
        let probe = source.probe();
        block.add_probe(source_label);
        Ok(probe)
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
        let link = graph.add_link("main/c", "main/a").unwrap();
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
        graph.add_link("main/c", "main/a").unwrap();
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
        graph.add_probe::<u32>("main/c").unwrap();
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
        graph.add_probe::<u32>("main/c").unwrap();
    }
}
