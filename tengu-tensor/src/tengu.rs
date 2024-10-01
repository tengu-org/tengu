use std::fmt::Display;
use std::rc::Rc;

use tengu_wgpu::{Device, WGPU};

use crate::{expression::Expression, graph::Graph, tensor::TensorBuilder, Result};

// Limit available underlying types.

pub trait IOType: StorageType + bytemuck::Pod {}

impl IOType for f32 {}
impl IOType for u32 {}
impl IOType for i32 {}

pub trait StorageType: Display + Clone + 'static {}

impl StorageType for f32 {}
impl StorageType for u32 {}
impl StorageType for i32 {}
impl StorageType for bool {}

// Tengu implementation.

pub struct Tengu {
    device: Device,
}

impl Tengu {
    pub async fn new() -> Result<Rc<Self>> {
        let device = WGPU::default_context().await?;
        Ok(Rc::new(Self { device }))
    }

    pub(crate) fn device<'a>(self: &'a Rc<Self>) -> &'a Device {
        &self.device
    }

    pub fn tensor(self: &Rc<Self>, shape: impl Into<Vec<usize>>) -> TensorBuilder {
        TensorBuilder::new(self, shape)
    }

    pub fn scalar<T: IOType>(self: &Rc<Self>, scalar: T) -> Expression<T> {
        Expression::Scalar(scalar)
    }

    pub fn graph<'a>(self: &'a Rc<Self>) -> Graph<'a> {
        Graph::new(self)
    }
}
