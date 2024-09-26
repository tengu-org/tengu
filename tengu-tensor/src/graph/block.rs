use itertools::Itertools;
use std::sync::Arc;

use super::Computation;
use crate::{expression::Expression, Emit, Tengu};

pub struct Block<T> {
    tengu: Arc<Tengu>,
    computations: Vec<Computation<T>>,
}

impl<T: 'static> Block<T> {
    pub fn new(tengu: Arc<Tengu>) -> Self {
        Self {
            tengu,
            computations: Vec::new(),
        }
    }

    pub fn add_computation(&mut self, expression: Expression<T>) -> &Computation<T> {
        self.computations
            .push(Computation::new(Arc::clone(&self.tengu), expression));
        self.computations
            .last()
            .expect("Should have a least one computation after adding a new one")
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

impl<T: 'static> Emit for Block<T> {
    fn emit(&self) -> String {
        let declaration = self.declaration();
        let body = self.body();
        format!("{declaration}\n{body}")
    }
}
