use std::sync::Arc;

use block::Block;
use computation::Computation;

use crate::{Emit, Tengu};

mod block;
mod computation;

pub struct Graph {
    tengu: Arc<Tengu>,
    blocks: Vec<Box<dyn Emit>>,
}

impl Graph {
    pub fn new(tengu: Arc<Tengu>) -> Self {
        Self {
            tengu,
            blocks: Vec::new(),
        }
    }

    pub fn add_block<T: 'static>(&mut self) -> &Block<T> {
        let block = Block::<T>::new(Arc::clone(&self.tengu));
        self.blocks.push(Box::new(block));
        self.blocks
            .last()
            .expect("Graph blocks should not be empty after inserting a new block")
            .downcast_ref()
            .expect("Added block should have correct type after downcasting")
    }
}
