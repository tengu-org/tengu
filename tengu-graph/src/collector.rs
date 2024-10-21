//! This module defines the `Collector` struct and associated functionality for collecting sources
//! within a computation graph. The collector is used to gather all sources required for a computation
//! and to ensure that only the necessary sources are processed.

use std::collections::{HashMap, HashSet};

use tengu_backend::Backend;

use crate::source::Source;

/// Struct representing a source collector for a computation graph.
pub struct Collector<'a, B: Backend> {
    filter: &'a HashSet<String>,
    sources: HashMap<&'a str, &'a dyn Source<B>>,
}

impl<'a, B: Backend + 'static> Collector<'a, B> {
    /// Creates a new `Collector` instance for the specified source labels.
    /// These labels represent tensors that have probes associated with them.
    ///
    /// # Parameters
    /// - `filter`: A reference to a set of source labels to collect.
    ///
    /// # Returns
    /// A new `Collector` instance.
    pub fn new(filter: &'a HashSet<String>) -> Self {
        Self {
            filter,
            sources: HashMap::new(),
        }
    }

    /// Adds a source to the collector.
    ///
    /// # Parameters
    /// - `source`: A reference to the source to add.
    pub fn add(&mut self, source: &'a dyn Source<B>) {
        if self.filter.contains(source.label()) {
            self.sources.entry(source.label()).or_insert(source);
        }
    }

    /// Returns an iterator over the sources in the collector.
    ///
    /// # Returns
    /// An iterator over the sources in the collector.
    pub fn sources(&'a self) -> impl Iterator<Item = &'a dyn Source<B>> {
        self.sources.values().copied()
    }
}
