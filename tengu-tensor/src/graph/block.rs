use indoc::formatdoc;
use itertools::Itertools;
use std::{any::Any, cell::OnceCell, rc::Rc};
use tengu_wgpu::Pipeline;

use super::Computation;
use crate::{Expression, Probe, Tengu, Tensor, WGSLType};

const WORKGROUP_SIZE: u32 = 64;
const GROUP: usize = 0;

// Block implementation

pub struct Block<T> {
    tengu: Rc<Tengu>,
    label: String,
    computations: Vec<Computation<T>>,
    pipeline: OnceCell<Pipeline>,
}

impl<T: WGSLType> Block<T> {
    pub fn new(tengu: &Rc<Tengu>, label: impl Into<String>) -> Self {
        Self {
            tengu: Rc::clone(tengu),
            label: label.into(),
            computations: Vec::new(),
            pipeline: OnceCell::new(),
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
        format!("{declaration}\n\n{body}")
    }

    fn declaration(&self) -> String {
        self.computations
            .iter()
            .flat_map(|computation| computation.nodes())
            .unique_by(|tensor| tensor.label())
            .enumerate()
            .map(|(binding, tensor)| tensor.declaration(GROUP, binding))
            .join("\n")
    }

    fn body(&self) -> String {
        formatdoc!(
            r"
            @compute
            @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                {}
            }}",
            self.computations.iter().map(|c| c.emit()).join("\n    "),
        )
    }

    fn pipeline(&self) -> &Pipeline {
        self.pipeline.get_or_init(|| {
            let shader = self.tengu.device().shader(self.label(), &self.emit());
            let buffers = self.nodes().map(|t| t.buffer());
            self.tengu
                .device()
                .layout()
                .add_entries(buffers)
                .pipeline(self.label())
                .build(shader)
        })
    }
}

// Traits

pub trait Compute {
    fn compute(&self) -> wgpu::CommandBuffer;
    fn probe<'a>(&'a self, block_label: &str, tensor_label: &str) -> Option<&'a Probe>;
    fn as_any(&mut self) -> &mut dyn Any;
}

impl<T: WGSLType + 'static> Compute for Block<T> {
    fn compute(&self) -> wgpu::CommandBuffer {
        let pipeline = self.pipeline();
        let mut encoder = self.tengu.device().compute(&self.label, |pass| {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, pipeline.bind_group(), &[]);
            pass.dispatch_workgroups((self.count() as u32 / WORKGROUP_SIZE) + 1, 1, 1);
        });
        self.nodes().for_each(|tensor| tensor.read(&mut encoder));
        encoder.finish()
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

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn builder_declaration() {
        let tengu = Tengu::new().await.unwrap();
        let a = tengu.tensor([2, 2]).with_label("a").init(&[1.0, 2.0, 3.0, 4.0]);
        let b = tengu.tensor([2, 2]).with_label("b").init(&[5.0, 6.0, 7.0, 8.0]);
        let mut graph = tengu.graph();
        let block = graph.add_block("addition").add_computation("out", a + b);
        let declaration = block.declaration();
        let declaration = declaration.lines().collect::<Vec<_>>();
        assert!(declaration.contains(&"@group(0) @binding(0) var<storage, read> a: array<f32>;"));
        assert!(declaration.contains(&"@group(0) @binding(1) var<storage, read> b: array<f32>;"));
        assert!(declaration.contains(&"@group(0) @binding(2) var<storage, read_write> out: array<f32>;"));
    }

    #[tokio::test]
    async fn builder_body() {
        let tengu = Tengu::new().await.unwrap();
        let a = tengu.tensor([2, 2]).with_label("a").empty::<f32>();
        let b = tengu.tensor([2, 2]).with_label("b").empty::<f32>();
        let mut graph = tengu.graph();
        let block = graph.add_block("addition").add_computation("out", a + b);
        assert_eq!(
            block.body(),
            indoc!(
                r"
                @compute
                @workgroup_size(64)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let idx = global_id.x;
                    out[idx] = (a[idx] + b[idx]);
                }"
            )
        );
    }
}
