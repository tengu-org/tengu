use std::fmt::Display;
use std::rc::Rc;

use tengu_wgpu::{Device, WGPU};

use crate::tensor::TensorBuilder;
use crate::{Expression, Graph, Result};

// Limit available underlying types.

pub trait WGSLType: bytemuck::Pod + Display {}

impl WGSLType for f32 {}
impl WGSLType for u32 {}
impl WGSLType for i32 {}

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

    pub fn scalar<T: WGSLType>(self: &Rc<Self>, scalar: T) -> Expression<T> {
        Expression::Scalar(scalar)
    }

    pub fn graph(self: &Rc<Self>) -> Graph {
        Graph::new(self)
    }
}
