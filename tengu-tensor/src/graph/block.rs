use itertools::Itertools;
use std::{any::Any, collections::HashMap, sync::Arc};
use tengu_wgpu::Buffer;

use super::descriptor::{Describe, Descriptor};
use super::Computation;
use crate::{expression::Expression, Tengu, Tensor};

pub struct Block<T> {
    tengu: Arc<Tengu>,
    computations: Vec<Computation<T>>,
}

impl<T: 'static> Block<T> {
    pub fn new(tengu: &Arc<Tengu>) -> Self {
        Self {
            tengu: Arc::clone(tengu),
            computations: Vec::new(),
        }
    }

    pub fn add_computation(&mut self, expression: Expression<T>) -> &mut Computation<T> {
        self.computations.push(Computation::new(&self.tengu, expression));
        self.computations
            .last_mut()
            .expect("Should have a least one computation after adding a new one")
    }

    pub fn count(&self) -> usize {
        self.computations.iter().map(|c| c.count()).max().unwrap_or_default()
    }

    pub fn sources(&self) -> Vec<&Buffer> {
        self.computations
            .iter()
            .flat_map(|c| c.sources().into_iter())
            .collect::<HashMap<&str, &Tensor<T>>>()
            .values()
            .map(|tensor| tensor.buffer())
            .collect()
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
}

// Describe trait implementation

impl<T: 'static> Describe for Block<T> {
    fn descriptor(&self) -> super::descriptor::Descriptor {
        Descriptor {
            count: self.count(),
            rep: self.emit(),
            buffers: self.sources(),
        }
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}
