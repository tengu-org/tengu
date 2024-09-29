use std::sync::Arc;

use tengu_wgpu::{Device, WGPU};

use crate::tensor::TensorBuilder;
use crate::{Graph, Result};

pub struct Tengu {
    device: Device,
}

impl Tengu {
    pub async fn new() -> Result<Arc<Self>> {
        let device = WGPU::default_context().await?;
        Ok(Arc::new(Self { device }))
    }

    pub(crate) fn device<'a>(self: &'a Arc<Self>) -> &'a Device {
        &self.device
    }

    pub fn tensor<T>(self: &Arc<Self>, shape: impl Into<Vec<usize>>) -> TensorBuilder<T> {
        TensorBuilder::new(self, shape)
    }

    pub fn graph(self: &Arc<Self>) -> Graph {
        Graph::new(self)
    }
}
