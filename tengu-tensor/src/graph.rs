use std::sync::Arc;

use block::Block;
use computation::Computation;
use downcast_rs::{impl_downcast, Downcast};
use tengu_wgpu::ComputePipeline;

use crate::{Emit, Tengu};

mod block;
mod computation;

// Block tratis

trait Count {
    fn count(&self) -> Option<usize>;
}

trait Compute: Emit + Count + Downcast {}

impl<T: 'static> Compute for T where T: Emit + Count {}

impl_downcast!(Compute);

// Graph implementation

pub struct Graph {
    tengu: Arc<Tengu>,
    blocks: Vec<Box<dyn Compute>>,
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
            .downcast_mut()
            .expect("Added block should have correct type after downcasting")
    }

    pub async fn compute(&self) {
        let shader_source = self.blocks.first().unwrap().emit();
        let shader_module = self.tengu.device().shader(&shader_source);
        let compute_pipeline = self.create_pipeline();
        let count = self.blocks.iter().flat_map(|b| b.count()).max().unwrap();
        self.tengu.device().compute(|encoder| {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline.build(shader_module));
            compute_pass.set_bind_group(0, compute_pipeline.bind_group(), &[]);
            compute_pass.dispatch_workgroups((count as u32 + 63) / 64, 1, 1); // Assumes workgroup_size is 64
        });
    }

    fn create_pipeline(&self) -> ComputePipeline {
        todo!();
    }
}
