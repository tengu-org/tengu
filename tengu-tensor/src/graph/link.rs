use tengu_backend::Backend;

use super::Graph;
use crate::expression::Source;
use crate::{Error, Result};

pub struct Link {
    from: String,
    to: String,
}

impl Link {
    pub fn new<B: Backend + 'static>(graph: &Graph<B>, from: impl Into<String>, to: impl Into<String>) -> Result<Self> {
        let from = from.into();
        let to = to.into();
        let from_source = graph.get_source(&from)?;
        let to_source = graph.get_source(&to)?;
        if !from_source.matches_with(to_source)? {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self { from, to })
    }

    pub fn realize<'a, B: Backend + 'static>(&self, graph: &'a Graph<B>) -> (&'a dyn Source<B>, &'a dyn Source<B>) {
        let from = graph.get_source(&self.from).expect("link from source should exist");
        let to = graph.get_source(&self.to).expect("link to source should exist");
        (from, to)
    }
}
