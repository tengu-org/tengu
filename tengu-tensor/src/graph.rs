use std::rc::Rc;

pub use block::Block;
use block::Compute;
pub use computation::Computation;

use crate::{Probe, Tengu, WGSLType};

mod block;
mod computation;

// Graph implementation

pub struct Graph {
    tengu: Rc<Tengu>,
    blocks: Vec<Box<dyn Compute>>,
}

impl Graph {
    pub fn new(tengu: &Rc<Tengu>) -> Self {
        Self {
            tengu: Rc::clone(tengu),
            blocks: Vec::new(),
        }
    }

    pub fn add_block<T: WGSLType + 'static>(&mut self, label: impl Into<String>) -> &mut Block<T> {
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

    pub fn step(&self) {
        let commands = self.blocks.iter().map(|block| block.compute());
        self.tengu.device().submit(commands);
    }
}
