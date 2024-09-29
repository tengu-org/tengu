use std::{collections::HashMap, sync::Arc};

use itertools::Itertools;

use crate::{Expression, Tengu, Tensor, WGSLType};

pub struct Computation<T> {
    expression: Expression<T>,
    output: Tensor<T>,
}

impl<T: WGSLType> Computation<T> {
    pub fn new(tengu: &Arc<Tengu>, label: impl Into<String>, expression: Expression<T>) -> Self {
        let output = tengu.tensor(expression.shape()).with_label(label).empty();
        Self { expression, output }
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

    pub(crate) fn nodes(&self) -> impl Iterator<Item = &Tensor<T>> {
        self.expression
            .inputs()
            .into_iter()
            .chain(std::iter::once(&self.output))
            .unique_by(|t| t.label())
    }

    pub fn emit(&self) -> String {
        format!("{} = {};", self.output.emit(), self.expression.emit())
    }

    pub fn count(&self) -> usize {
        self.expression.count()
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn computation_builder() {
        let tengu = Tengu::new().await.unwrap();
        let a = tengu.tensor([2, 2]).init(&[1.0, 2.0, 3.0, 4.0]);
        let b = tengu.tensor([2, 2]).init(&[5.0, 6.0, 7.0, 8.0]);
        let computation = Computation::new(&tengu, "c", a + b);
        assert_eq!(computation.count(), 4);
    }

    #[tokio::test]
    async fn computation_declaration() {
        let tengu = Tengu::new().await.unwrap();
        let a = tengu.tensor([2, 2]).with_label("a").init(&[1.0, 2.0, 3.0, 4.0]);
        let b = tengu.tensor([2, 2]).with_label("b").init(&[5.0, 6.0, 7.0, 8.0]);
        let computation = Computation::new(&tengu, "c", a + b);
        let declarations = computation.declarations(1);
        assert_eq!(declarations.len(), 3);
        assert_eq!(
            declarations.get("a").unwrap(),
            "@group(1) @binding(0) var<storage, read_write> a: array<f32>;"
        );
        assert_eq!(
            declarations.get("b").unwrap(),
            "@group(1) @binding(1) var<storage, read_write> b: array<f32>;"
        );
        assert_eq!(
            declarations.get("c").unwrap(),
            "@group(1) @binding(2) var<storage, read_write> c: array<f32>;"
        );
    }

    #[tokio::test]
    async fn computation_emit() {
        let tengu = Tengu::new().await.unwrap();
        let a = tengu.tensor([2, 2]).with_label("a").empty::<f32>();
        let b = tengu.tensor([2, 2]).with_label("b").empty::<f32>();
        let computation = Computation::new(&tengu, "c", a + b);
        assert_eq!(computation.emit(), "c[idx] = (a[idx] + b[idx]);");
    }
}
