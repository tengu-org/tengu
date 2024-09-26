use std::sync::Arc;

use block::Block;

use crate::{expression::Expression, Computation, Tengu};

mod block;

pub struct Graph<'a> {
    tengu: Arc<Tengu>,
    blocks: Vec<Box<dyn Computation + 'a>>,
}

impl<'a> Graph<'a> {
    pub fn new(tengu: Arc<Tengu>) -> Self {
        Self {
            tengu,
            blocks: Vec::new(),
        }
    }

    pub fn add_block<T: 'a>(&mut self, expression: Expression<T>) -> &Box<dyn Computation + 'a> {
        let block = Block::new(Arc::clone(&self.tengu), expression);
        self.blocks.push(Box::new(block));
        self.blocks
            .last()
            .expect("Graph blocks should not be mpty after inserting a new block")
    }
}
