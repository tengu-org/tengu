use indoc::formatdoc;
use itertools::Itertools;
use std::{cell::OnceCell, rc::Rc};
use tengu_wgpu::Pipeline;

use super::computation::Computation;
use crate::expression::traits::{Node, Source};
use crate::expression::Expression;
use crate::probe::Probe;
use crate::visitor::Visitor;
use crate::{IOType, Tengu};

const WORKGROUP_SIZE: u32 = 64;
const GROUP: usize = 0;

// Block implementation

pub struct Block<'a> {
    tengu: Rc<Tengu>,
    label: String,
    computations: Vec<Box<dyn Node>>,
    visitor: OnceCell<Visitor<'a>>,
    pipeline: OnceCell<Pipeline>,
    shader: OnceCell<String>,
}

// Public interface

impl<'a> Block<'a> {
    pub fn new(tengu: &Rc<Tengu>, label: impl Into<String>) -> Self {
        Self {
            tengu: Rc::clone(tengu),
            label: label.into(),
            computations: Vec::new(),
            visitor: OnceCell::new(),
            pipeline: OnceCell::new(),
            shader: OnceCell::new(),
        }
    }

    pub fn add_computation<T: IOType>(&mut self, label: impl Into<String>, expr: Expression<T>) -> &mut Self {
        let computation = Computation::new(&self.tengu, label, expr);
        self.computations.push(Box::new(computation));
        self
    }

    pub fn compute(&'a self) -> wgpu::CommandBuffer {
        let pipeline = self.pipeline();
        let mut encoder = self.tengu.device().encoder(&self.label).pass(&self.label, |pass| {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, pipeline.bind_group(), &[]);
            pass.dispatch_workgroups((self.count() as u32 / WORKGROUP_SIZE) + 1, 1, 1);
        });
        self.visitor().sources().for_each(|tensor| tensor.read(&mut encoder));
        encoder.finish()
    }

    pub fn probe(&'a self, block_label: &str, tensor_label: &str) -> Option<&'a Probe> {
        (self.label() == block_label)
            .then(|| self.visitor().get(tensor_label).map(|tensor| tensor.probe()))
            .flatten()
    }

    pub fn source(&'a self, label: &str) -> Option<&'a dyn Source> {
        self.visitor().get(label)
    }
}

// Private methods

impl<'a> Block<'a> {
    fn count(&self) -> usize {
        self.computations.iter().map(|c| c.count()).max().unwrap_or_default()
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn shader(&'a self) -> &str {
        self.shader.get_or_init(|| {
            let declaration = self.declaration();
            let body = self.body();
            format!("{declaration}\n\n{body}")
        })
    }

    fn declaration(&'a self) -> String {
        self.visitor()
            .sources()
            .enumerate()
            .map(|(binding, source)| source.declaration(GROUP, binding))
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
            self.computations.iter().map(|node| node.emit()).join("\n    "),
        )
    }

    fn visitor(&'a self) -> &'a Visitor<'a> {
        self.visitor.get_or_init(|| {
            let mut visitor = Visitor::new();
            self.computations.iter().for_each(|node| node.visit(&mut visitor));
            visitor
        })
    }

    fn pipeline(&'a self) -> &Pipeline {
        self.pipeline.get_or_init(|| {
            let shader = self.tengu.device().shader(self.label(), self.shader());
            let buffers = self.visitor().sources().map(|source| source.buffer());
            self.tengu
                .device()
                .layout()
                .add_entries(buffers)
                .pipeline(self.label())
                .build(shader)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use regex::RegexSet;

    #[tokio::test]
    async fn builder_declaration() {
        let tengu = Tengu::new().await.unwrap();
        let a = tengu.tensor([2, 2]).label("a").init(&[1.0, 2.0, 3.0, 4.0]);
        let b = tengu.tensor([2, 2]).label("b").init(&[5.0, 6.0, 7.0, 8.0]);
        let mut graph = tengu.graph();
        let block = graph.add_block("addition").unwrap().add_computation("out", a + b);
        let declaration = block.declaration();
        let declarations = declaration.lines().collect::<Vec<_>>();
        let re = RegexSet::new([
            r"@group\(0\) @binding\(\d+\) var<storage, read> a: array<f32>;",
            r"@group\(0\) @binding\(\d+\) var<storage, read> b: array<f32>;",
            r"@group\(0\) @binding\(\d+\) var<storage, read_write> out: array<f32>;",
        ])
        .unwrap();
        for declaration in declarations {
            println!("{:?}", declaration);
            assert!(re.is_match(declaration));
        }
    }

    #[tokio::test]
    async fn builder_body() {
        let tengu = Tengu::new().await.unwrap();
        let a = tengu.tensor([2, 2]).label("a").zero::<f32>();
        let b = tengu.tensor([2, 2]).label("b").zero::<f32>();
        let mut graph = tengu.graph();
        let block = graph.add_block("addition").unwrap().add_computation("out", a + b);
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
