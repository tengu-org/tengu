use std::fmt::Display;
use std::rc::Rc;

use tengu_wgpu::{Device, WGPU};

use crate::{expression::Expression, graph::Graph, tensor::TensorBuilder, Result};

// Limit available underlying types.

pub trait PodType: bytemuck::Pod + Display {}

impl PodType for f32 {}
impl PodType for u32 {}
impl PodType for i32 {}

pub trait WGSLType: Display + Clone + 'static {}

impl WGSLType for f32 {}
impl WGSLType for u32 {}
impl WGSLType for i32 {}
impl WGSLType for bool {}

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

    pub fn scalar<T: PodType>(self: &Rc<Self>, scalar: T) -> Expression<T> {
        Expression::Scalar(scalar)
    }

    pub fn graph<'a>(self: &'a Rc<Self>) -> Graph<'a> {
        Graph::new(self)
    }
}
