use std::sync::Arc;

use block::Block;
use computation::Computation;
use descriptor::{Describe, Descriptor};
use tengu_wgpu::Pipeline;

use crate::Tengu;

mod block;
mod computation;
mod descriptor;

const WORKGROUP_SIZE: u32 = 64;

// Graph implementation

pub struct Graph {
    tengu: Arc<Tengu>,
    blocks: Vec<Box<dyn Describe>>,
}

impl Graph {
    pub fn new(tengu: &Arc<Tengu>) -> Self {
        Self {
            tengu: Arc::clone(tengu),
            blocks: Vec::new(),
        }
    }

    pub fn add_block<T: 'static>(&mut self) -> &mut Block<T> {
        let block = Block::<T>::new(&self.tengu);
        self.blocks.push(Box::new(block));
        self.blocks
            .last_mut()
            .expect("Graph blocks should not be empty after inserting a new block")
            .as_any()
            .downcast_mut()
            .expect("Added block should have correct type after downcasting")
    }

    pub async fn compute(&self) {
        let block = self.blocks.first().unwrap().descriptor();
        self.compute_block(block).await;
    }

    pub async fn compute_block(&self, descriptor: Descriptor<'_>) {
        let count = descriptor.count;
        let pipeline = self.create_pipeline(descriptor);
        self.tengu.device().compute(|encoder| {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, pipeline.bind_group(), &[]);
            compute_pass.dispatch_workgroups((count as u32 / WORKGROUP_SIZE) + 1, 1, 1);
        });
    }

    fn create_pipeline(&self, descriptor: Descriptor) -> Pipeline {
        let shader = self.tengu.device().shader(&descriptor.rep);
        self.tengu
            .device()
            .layout()
            .add_entries(descriptor.buffers)
            .pipeline()
            .build(shader)
    }
}
