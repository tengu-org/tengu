//! Module for managing links between sources in the Tengu tensor computation framework.
//!
//! This module defines the `Link` struct and associated methods for creating and managing
//! links between sources in a computational graph. Links are used to establish connections
//! between different nodes in the graph, ensuring that data flows correctly between them.

use tengu_backend::Backend;

use super::Graph;
use crate::expression::Source;
use crate::{Error, Result};

/// A struct representing a link between two tensors in a computational graph.
///
/// The `Link` struct holds the identifiers of the source and destination tensors and provides
/// methods to create, access, and realize the link by turning tensors labels to source objects.
pub struct Link {
    from: String,
    to: String,
}

impl Link {
    /// Creates a new `Link` instance between two tensors in the graph.
    ///
    /// # Type Parameters
    /// - `B`: The backend type.
    ///
    /// # Parameters
    /// - `graph`: A reference to the computational graph.
    /// - `from`: The identifier of the source tensor, in "block/tensor" format.
    /// - `to`: The identifier of the destination tensor, in "block/tensor" format.
    ///
    /// # Returns
    /// A result containing the created `Link` instance or an error if the sources do not match.
    ///
    /// # Errors
    /// Returns `Error::ShapeMismatch` if the shapes of the sources do not match.
    pub fn new<B: Backend + 'static>(graph: &Graph<B>, from: impl Into<String>, to: impl Into<String>) -> Result<Self> {
        let from = from.into();
        let to = to.into();
        let from_source = graph.get_source(&from)?;
        let to_source = graph.get_source(&to)?;
        if !from_source.matches_to(to_source)? {
            return Err(Error::ShapeMismatch);
        }
        Ok(Self { from, to })
    }

    /// Returns the identifier of the source tensor, in "block/tensor" format.
    ///
    /// # Returns
    /// A reference to the identifier string of the source node.
    pub fn from(&self) -> &str {
        &self.from
    }

    /// Returns the identifier of the destination tensor, in "block/tensor" format.
    ///
    /// # Returns
    /// A reference to the identifier string of the destination tensor.
    pub fn to(&self) -> &str {
        &self.to
    }

    /// Realizes the link by retrieving the source and destination nodes corresponding to tensor
    /// labels from the graph.
    ///
    /// # Type Parameters
    /// - `B`: The backend type.
    ///
    /// # Parameters
    /// - `graph`: A reference to the computational graph.
    ///
    /// # Returns
    /// A tuple containing references to the source and destination nodes.
    ///
    /// # Panics
    /// Panics if the source or destination nodes do not exist in the graph.
    pub(crate) fn realize<'a, B: Backend + 'static>(
        &self,
        graph: &'a Graph<B>,
    ) -> (&'a dyn Source<B>, &'a dyn Source<B>) {
        let from = graph.get_source(&self.from).expect("link from source should exist");
        let to = graph.get_source(&self.to).expect("link to source should exist");
        (from, to)
    }
}
