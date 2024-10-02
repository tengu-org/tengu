use std::rc::Rc;

use crate::builder::Builder;
use crate::expression::{Expression, Shape};
use crate::graph::Graph;
use crate::Result;
use crate::WGPUBackend;
use tengu_backend::{Backend, IOType};

pub struct Tengu<B> {
    backend: Rc<B>,
}

impl<B: Backend + 'static> Tengu<B> {
    pub async fn new() -> Result<Rc<Self>> {
        let backend = B::new().await?;
        Ok(Rc::new(Self { backend }))
    }

    pub fn backend(self: &Rc<Self>) -> &Rc<B> {
        &self.backend
    }

    pub fn tensor(self: &Rc<Self>, shape: impl Into<Vec<usize>>) -> Builder<B> {
        Builder::new(&self.backend, shape)
    }

    pub fn like<T: IOType>(self: &Rc<Self>, expr: &Expression<T, B>) -> Builder<B> {
        Builder::new(&self.backend, expr.shape())
    }

    pub fn scalar<T: IOType>(self: &Rc<Self>, scalar: T) -> Expression<T, B> {
        Expression::Scalar(scalar)
    }

    pub fn graph(self: &Rc<Self>) -> Graph<B> {
        Graph::new(self)
    }
}

impl Tengu<WGPUBackend> {
    pub async fn wgpu() -> Result<Rc<Self>> {
        Tengu::new().await
    }
}
