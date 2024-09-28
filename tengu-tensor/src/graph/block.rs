use itertools::Itertools;
use std::{any::Any, sync::Arc};
use tengu_wgpu::Pipeline;

use super::Computation;
use crate::{Expression, Probe, Tengu, Tensor};

const WORKGROUP_SIZE: u32 = 64;

// Block implementation

pub struct Block<T> {
    tengu: Arc<Tengu>,
    label: String,
    computations: Vec<Computation<T>>,
}

impl<T: 'static> Block<T> {
    pub fn new(tengu: &Arc<Tengu>, label: impl Into<String>) -> Self {
        Self {
            tengu: Arc::clone(tengu),
            label: label.into(),
            computations: Vec::new(),
        }
    }

    pub fn add_computation(&mut self, label: impl Into<String>, expression: Expression<T>) -> &mut Self {
        let computation = Computation::new(&self.tengu, label, expression);
        self.computations.push(computation);
        self
    }

    pub fn count(&self) -> usize {
        self.computations.iter().map(|c| c.count()).max().unwrap_or_default()
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub(crate) fn nodes(&self) -> impl Iterator<Item = &Tensor<T>> {
        self.computations
            .iter()
            .flat_map(|c| c.nodes())
            .unique_by(|t| t.label())
    }

    pub fn emit(&self) -> String {
        let declaration = self.declaration();
        let body = self.body();
        format!("{declaration}\n{body}")
    }

    fn declaration(&self) -> String {
        let group = 0; // TODO: Should we increment this somewhere?
        self.computations
            .iter()
            .map(|computation| computation.declarations(group))
            .concat()
            .values()
            .join("\n")
    }

    fn body(&self) -> String {
        format!(
            r"
            @compute
            @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                {}
            }}",
            self.computations.iter().map(|c| c.emit()).join("\n"),
        )
    }

    fn create_pipeline(&self) -> Pipeline {
        let shader = self.tengu.device().shader(&self.emit());
        let buffers = self.nodes().map(|t| t.buffer());
        self.tengu
            .device()
            .layout()
            .add_entries(buffers)
            .pipeline()
            .build(shader)
    }
}

// Traits

pub trait Compute {
    fn compute(&self);
    fn probe<'a>(&'a self, block_label: &str, tensor_label: &str) -> Option<&'a Probe>;
    fn as_any(&mut self) -> &mut dyn Any;
}

impl<T: 'static> Compute for Block<T> {
    fn compute(&self) {
        let pipeline = self.create_pipeline();
        self.tengu.device().compute(|encoder| {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, pipeline.bind_group(), &[]);
            compute_pass.dispatch_workgroups((self.count() as u32 / WORKGROUP_SIZE) + 1, 1, 1);
        });
    }

    fn probe<'a>(&'a self, block_label: &str, tensor_label: &str) -> Option<&'a Probe> {
        (self.label() == block_label).then(|| {
            self.nodes()
                .find(|tensor| tensor.label() == tensor_label)
                .map(|tensor| tensor.probe())
        })?
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}
