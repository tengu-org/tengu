use std::{collections::HashMap, sync::Arc};

use crate::{Expression, Probe, Tengu, Tensor};

pub struct Computation<T> {
    tengu: Arc<Tengu>,
    expression: Expression<T>,
    output: Tensor<T>,
    probe: Option<Probe<T>>,
}

impl<T: 'static> Computation<T> {
    pub fn new(tengu: &Arc<Tengu>, expression: Expression<T>) -> Self {
        let output = tengu.tensor(expression.shape()).empty();
        Self {
            tengu: Arc::clone(tengu),
            expression,
            output,
            probe: None,
        }
    }

    pub fn with_label(tengu: &Arc<Tengu>, expression: Expression<T>, label: impl Into<String>) -> Self {
        let output = tengu.tensor(expression.shape()).with_label(label).empty();
        Self {
            tengu: Arc::clone(tengu),
            expression,
            output,
            probe: None,
        }
    }

    pub fn declarations(&self, group: usize) -> HashMap<&str, String> {
        self.expression
            .inputs()
            .into_iter()
            .chain(std::iter::once(&self.output))
            .enumerate()
            .map(|(binding, tensor)| (tensor.label(), tensor.declaration(group, binding)))
            .collect()
    }

    pub fn sources(&self) -> HashMap<&str, &Tensor<T>> {
        self.expression
            .inputs()
            .into_iter()
            .chain(std::iter::once(&self.output))
            .map(|tensor| (tensor.label(), tensor))
            .collect()
    }
    pub fn emit(&self) -> String {
        format!("{} = {};", self.output.emit(), self.expression.emit())
    }

    pub fn probe(&mut self) -> &Probe<T> {
        let probe = Probe::new(&self.tengu, self.output.count());
        self.probe = Some(probe);
        self.probe.as_ref().expect("Should have probe after setting one")
    }

    pub fn count(&self) -> usize {
        self.expression.count()
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn computation_builder() {
        let tengu = Tengu::new().await.unwrap();
        let a = tengu.tensor([2, 2]).init(&[1.0, 2.0, 3.0, 4.0]);
        let b = tengu.tensor([2, 2]).init(&[5.0, 6.0, 7.0, 8.0]);
        let computation = Computation::new(&tengu, a + b);
        assert_eq!(computation.count(), 4);
    }

    #[tokio::test]
    async fn computation_declaration() {
        let tengu = Tengu::new().await.unwrap();
        let a = tengu.tensor([2, 2]).with_label("a").init(&[1.0, 2.0, 3.0, 4.0]);
        let b = tengu.tensor([2, 2]).with_label("b").init(&[5.0, 6.0, 7.0, 8.0]);
        let computation = Computation::with_label(&tengu, a + b, "c");
        let declarations = computation.declarations(1);
        assert_eq!(declarations.len(), 3);
        assert_eq!(
            declarations.get("a").unwrap(),
            "@group(1) @binding(0) var<storage, read> a: array<f32>"
        );
        assert_eq!(
            declarations.get("b").unwrap(),
            "@group(1) @binding(1) var<storage, read> b: array<f32>"
        );
        assert_eq!(
            declarations.get("3").unwrap(),
            "@group(1) @binding(1) var<storage, read_write> c: array<f32>"
        );
    }

    #[tokio::test]
    async fn computation_emit() {
        let tengu = Tengu::new().await.unwrap();
        let a = tengu.tensor([2, 2]).with_label("a").init(&[1.0, 2.0, 3.0, 4.0]);
        let b = tengu.tensor([2, 2]).with_label("b").init(&[5.0, 6.0, 7.0, 8.0]);
        let computation = Computation::with_label(&tengu, a + b, "c");
        assert_eq!(computation.emit(), "c[idx] = (a[idx] + b[idx]);");
    }
}
