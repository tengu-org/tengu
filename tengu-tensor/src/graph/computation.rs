use std::{collections::HashMap, sync::Arc};

use crate::{Emit, Expression, Probable, Probe, Tengu, Tensor};

pub struct Computation<T> {
    tengu: Arc<Tengu>,
    expression: Expression<T>,
    output: Tensor<T>,
    probe: Option<Probe<T>>,
}

impl<T: 'static> Computation<T> {
    pub fn new(tengu: Arc<Tengu>, expression: Expression<T>) -> Self {
        let output = tengu.tensor(expression.shape()).empty();
        Self {
            tengu,
            expression,
            output,
            probe: None,
        }
    }

    pub fn declarations(&self, group: usize) -> HashMap<String, String> {
        self.expression
            .inputs()
            .into_iter()
            .chain(std::iter::once(&self.output))
            .enumerate()
            .map(|(binding, tensor)| (tensor.emit(), tensor.declaration(group, binding)))
            .collect()
    }
}

impl<T: 'static> Emit for Computation<T> {
    fn emit(&self) -> String {
        format!("{} = {}", self.output.emit(), self.expression.emit())
    }
}

impl<T> Probable<T> for Computation<T> {
    fn probe(&mut self) -> &Probe<T> {
        let probe = Probe::new(Arc::clone(&self.tengu), self.output.count());
        self.probe = Some(probe);
        self.probe.as_ref().expect("Should have probe after setting one")
    }
}
