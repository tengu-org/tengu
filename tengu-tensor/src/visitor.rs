use std::collections::HashMap;

use crate::frontend::Source;
use crate::tensor::Tensor;

pub struct Visitor<'a> {
    sources: HashMap<&'a str, &'a dyn Source>,
}

impl<'a> Visitor<'a> {
    pub fn new() -> Self {
        Self {
            sources: HashMap::new(),
        }
    }

    pub fn add<T>(&mut self, source: &'a Tensor<T>) {
        self.sources.insert(source.label(), source);
    }

    pub fn get(&'a self, label: &str) -> Option<&'a dyn Source> {
        self.sources.get(label).copied()
    }

    pub fn sources(&'a self) -> impl Iterator<Item = &'a dyn Source> {
        self.sources.values().copied()
    }
}
