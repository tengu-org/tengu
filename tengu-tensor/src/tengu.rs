//! Module for initializing and managing the Tengu tensor computation framework.
//!
//! This module defines the `Tengu` struct and associated methods for creating and managing
//! tensors, scalar expressions, and computational graphs. It provides an interface for
//! initializing the framework with a specified backend and includes specialized methods
//! for working with tensors and scalars. This is what user sees first when they start interacting
//! with Tengu.

use std::sync::Arc;
use tengu_backend::{Backend, IOType};

use crate::builder::Builder;
use crate::expression::{Expression, Shape};
use crate::graph::Graph;
use crate::Result;
use crate::WGPUBackend;

/// Main struct for the Tengu tensor computation framework.
///
/// The `Tengu` struct provides methods for managing tensors, expressions,
/// and computational graphs using a specified backend.
pub struct Tengu<B> {
    backend: Arc<B>,
}

impl<B: Backend + 'static> Tengu<B> {
    /// Creates a new instance of the Tengu framework with the specified backend.
    ///
    /// # Returns
    /// A result containing a reference-counted `Tengu` instance or an error.
    pub async fn new() -> Result<Arc<Self>> {
        let backend = B::new().await?;
        Ok(Arc::new(Self { backend }))
    }

    /// Returns a reference to the backend used by this Tengu instance.
    ///
    /// # Returns
    /// A reference to the backend.
    pub(crate) fn backend(self: &Arc<Self>) -> &Arc<B> {
        &self.backend
    }

    /// Creates a new tensor builder with the specified shape.
    ///
    /// # Parameters
    /// - `shape`: The shape of the tensor.
    ///
    /// # Returns
    /// A `Builder` instance for creating the tensor.
    pub fn tensor(self: &Arc<Self>, shape: impl Into<Vec<usize>>) -> Builder<B> {
        Builder::new(&self.backend, shape)
    }
    /// Creates a new tensor builder with the same shape as the specified expression.
    ///
    /// # Parameters
    /// - `expr`: An expression to match the shape.
    ///
    /// # Returns
    /// A `Builder` instance for creating the tensor.
    pub fn like<T: IOType>(self: &Arc<Self>, expr: &Expression<T, B>) -> Builder<B> {
        Builder::new(&self.backend, expr.shape())
    }

    /// Creates a scalar expression.
    ///
    /// # Parameters
    /// - `scalar`: The scalar value.
    ///
    /// # Returns
    /// An `Expression` representing the scalar.
    pub fn scalar<T: IOType>(self: &Arc<Self>, scalar: T) -> Expression<T, B> {
        Expression::Scalar(scalar)
    }

    /// Creates a new computational graph.
    ///
    /// # Returns
    /// A `Graph` instance for managing computations.
    pub fn graph(self: &Arc<Self>) -> Graph<B> {
        Graph::new(self)
    }
}

impl Tengu<WGPUBackend> {
    /// Creates a new instance of the Tengu framework with the WGPU backend.
    ///
    /// # Returns
    /// A result containing a reference-counted `Tengu` instance or an error.
    pub async fn wgpu() -> Result<Arc<Self>> {
        Tengu::new().await
    }
}
