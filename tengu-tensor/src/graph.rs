use std::sync::Arc;

pub use block::Block;
use block::Compute;
pub use computation::Computation;

use crate::{Probe, Tengu};

mod block;
mod computation;

// Graph implementation

pub struct Graph {
    tengu: Arc<Tengu>,
    blocks: Vec<Box<dyn Compute>>,
}

impl Graph {
    pub fn new(tengu: &Arc<Tengu>) -> Self {
        Self {
            tengu: Arc::clone(tengu),
            blocks: Vec::new(),
        }
    }

    pub fn add_block<T: 'static>(&mut self, label: impl Into<String>) -> &mut Block<T> {
        let block = Block::<T>::new(&self.tengu, label);
        self.blocks.push(Box::new(block));
        self.blocks
            .last_mut()
            .expect("Graph blocks should not be empty after inserting a new block")
            .as_any()
            .downcast_mut::<Block<T>>()
            .expect("Added block should have correct type after downcasting")
    }

    pub fn probe<'a>(&'a self, block_label: &str, tensor_label: &str) -> Option<&'a Probe> {
        self.blocks
            .iter()
            .flat_map(|block| block.probe(block_label, tensor_label))
            .next()
    }

    pub fn compute(&self) {
        for block in &self.blocks {
            block.compute();
        }
    }
}
