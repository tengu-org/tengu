use std::cell::RefCell;
use std::{collections::HashMap, future::Future, rc::Rc};

use crate::frontend::Source;
use crate::probe::Probe;
use crate::tengu::Tengu;
use crate::{Error, Result};
use block::Block;
use link::Link;

mod block;
mod computation;
mod link;

// Graph implementation

pub struct Graph<'a> {
    tengu: Rc<Tengu>,
    blocks: HashMap<String, Block<'a>>,
    links: RefCell<Vec<Link<'a>>>,
}

impl<'a> Graph<'a> {
    pub fn new(tengu: &Rc<Tengu>) -> Self {
        Self {
            tengu: Rc::clone(tengu),
            blocks: HashMap::new(),
            links: Vec::new().into(),
        }
    }

    pub fn add_block(&mut self, label: impl Into<String>) -> Result<&mut Block<'a>> {
        let label = label.into();
        if self.blocks.contains_key(&label) {
            return Err(Error::BlockAlreadyExists(label));
        }
        Ok(self
            .blocks
            .entry(label.clone())
            .or_insert(Block::new(&self.tengu, label.clone())))
    }

    pub fn link_one(&'a self, from: &str, to: &str) -> Result<()> {
        let from = self.get_source(from)?;
        let to = self.get_source(to)?;
        let link = Link::new(from, vec![to]);
        self.links.borrow_mut().push(link);
        Ok(())
    }

    pub fn probe(&'a self, path: &str) -> Result<&'a Probe> {
        Ok(self.get_source(path)?.probe())
    }

    pub fn step(&'a self) {
        let block_commands = self.blocks.values().map(|block| block.compute());
        let links_commands = self.tengu.device().compute("links", |encoder| {
            for link in self.links.borrow().iter() {
                link.compute(encoder);
            }
        });
        self.tengu
            .device()
            .submit(block_commands.chain(std::iter::once(links_commands)));
    }

    pub fn compute(&'a self, times: usize) {
        for _ in 0..times {
            self.step();
        }
    }

    pub async fn process<Fut>(&'a self, times: usize, call: impl Fn() -> Fut)
    where
        Fut: Future,
    {
        for _ in 0..times {
            self.step();
            call().await;
        }
    }

    pub async fn process_while<Fut>(&'a self, times: usize, call: impl Fn() -> Fut)
    where
        Fut: Future<Output = bool>,
    {
        for _ in 0..times {
            self.step();
            if !call().await {
                break;
            }
        }
    }

    fn get_source(&'a self, path: &str) -> Result<&'a dyn Source> {
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
