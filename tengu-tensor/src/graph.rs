use std::{collections::HashMap, rc::Rc};

use crate::expression::Source;
use crate::probe::Probe;
use crate::tensor::Tensor;
use crate::{Error, Result, Tengu};
use as_any::Downcast;
use block::Block;
use link::Link;
use tengu_backend::{Backend, StorageType};

mod block;
mod computation;
mod link;

pub struct Graph<B: Backend> {
    tengu: Rc<Tengu<B>>,
    blocks: HashMap<String, Block<B>>,
    links: Vec<Link>,
}

// Construction interface

impl<B: Backend + 'static> Graph<B> {
    pub fn new(tengu: &Rc<Tengu<B>>) -> Self {
        Self {
            tengu: Rc::clone(tengu),
            blocks: HashMap::new(),
            links: Vec::new(),
        }
    }

    pub fn add_block(&mut self, label: impl Into<String>) -> Result<&mut Block<B>> {
        let label = label.into();
        if self.blocks.contains_key(&label) {
            return Err(Error::BlockAlreadyExists(label));
        }
        Ok(self.blocks.entry(label).or_insert(Block::new(&self.tengu)))
    }

    pub fn link(&mut self, from: impl Into<String>, to: impl Into<String>) -> Result<()> {
        let link = Link::new(self, from, to)?;
        self.links.push(link);
        Ok(())
    }

    pub fn probe<T: StorageType>(&self, path: &str) -> Result<Probe<T, B>> {
        let source = self
            .get_source(path)?
            .downcast_ref::<Tensor<T, B>>()
            .ok_or_else(|| Error::TypeMismatch)?;
        Ok(source.probe())
    }

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

// Computation interface

impl<B: Backend + 'static> Graph<B> {
    pub fn compute(&self, times: usize) {
        let processors: Vec<_> = self.blocks.values().map(|block| block.processor()).collect();
        let links: Vec<_> = self.links.iter().map(|link| link.realize(self)).collect();
        let mut linker = self.tengu.backend().linker();
        for _ in 0..times {
            self.compute_blocks(&processors);
            self.compute_readout(&processors);
            self.compute_links(&mut linker, &links);
        }
    }

    pub fn process<F>(&self, times: usize, mut call: F)
    where
        F: FnMut(),
    {
        let processors: Vec<_> = self.blocks.values().map(|block| block.processor()).collect();
        let links: Vec<_> = self.links.iter().map(|link| link.realize(self)).collect();
        let mut linker = self.tengu.backend().linker();
        for _ in 0..times {
            self.compute_blocks(&processors);
            self.compute_readout(&processors);
            self.compute_links(&mut linker, &links);
            call();
        }
    }

    pub fn process_while<F>(&self, times: usize, mut call: F)
    where
        F: FnMut() -> bool,
    {
        let processors: Vec<_> = self.blocks.values().map(|block| block.processor()).collect();
        let links: Vec<_> = self.links.iter().map(|link| link.realize(self)).collect();
        let mut linker = self.tengu.backend().linker();
        for _ in 0..times {
            self.compute_blocks(&processors);
            self.compute_readout(&processors);
            self.compute_links(&mut linker, &links);
            if !call() {
                break;
            }
        }
    }

    fn compute_blocks(&self, processors: &Vec<B::Processor<'_>>) {
        self.tengu.backend().compute("compute", |mut compute| {
            for (block, processor) in self.blocks.values().zip(processors) {
                block.compute(&mut compute, processor);
            }
        });
    }

    fn compute_readout(&self, processors: &Vec<B::Processor<'_>>) {
        self.tengu.backend().readout("readout", |mut readout| {
            for (block, processor) in self.blocks.values().zip(processors) {
                block.readout(&mut readout, processor);
            }
        });
    }

    fn compute_links(&self, linker: &mut B::Linker, links: &Vec<(&dyn Source<B>, &dyn Source<B>)>) {
        for (from, to) in links {
            from.copy_link(*to, linker).expect("link endpoints should match");
        }
    }
}
