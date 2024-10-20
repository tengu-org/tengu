use std::collections::{HashMap, HashSet};

use tengu_backend::Backend;

use crate::source::Source;

pub struct Collector<'a, B: Backend> {
    filter: &'a HashSet<String>,
    sources: HashMap<&'a str, &'a dyn Source<B>>,
}

impl<'a, B: Backend + 'static> Collector<'a, B> {
    pub fn new(filter: &'a HashSet<String>) -> Self {
        Self {
            filter,
            sources: HashMap::new(),
        }
    }

    pub fn add(&mut self, source: &'a dyn Source<B>) {
        if self.filter.contains(source.label()) {
            self.sources.entry(source.label()).or_insert(source);
        }
    }

    pub fn sources(&'a self) -> impl Iterator<Item = &'a dyn Source<B>> {
        self.sources.values().copied()
    }
}
