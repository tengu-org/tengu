use std::rc::Rc;

use crate::{probe::Probe, Tengu};
use block::Block;

mod block;
mod computation;

// Graph implementation

pub struct Graph<'a> {
    tengu: Rc<Tengu>,
    blocks: Vec<Block<'a>>,
}

impl<'a> Graph<'a> {
    pub fn new(tengu: &Rc<Tengu>) -> Self {
        Self {
            tengu: Rc::clone(tengu),
            blocks: Vec::new(),
        }
    }

    pub fn add_block(&mut self, label: impl Into<String>) -> &mut Block<'a> {
        self.blocks.push(Block::new(&self.tengu, label));
        self.blocks
            .last_mut()
            .expect("Graph blocks should not be empty after inserting a new block")
    }

    pub fn probe(&'a self, block_label: &str, tensor_label: &str) -> Option<&'a Probe> {
        self.blocks
            .iter()
            .flat_map(|block| block.probe(block_label, tensor_label))
            .next()
    }

    pub fn step(&'a self) {
        let commands = self.blocks.iter().map(|block| block.compute());
        self.tengu.device().submit(commands);
    }
}
