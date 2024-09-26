use std::sync::Arc;

use add::AddExpression;

use crate::probe::impl_probable;
use crate::{Emit, Probable, Probe, Tengu, Tensor};

pub mod add;

pub enum Variant<T> {
    Tensor(Tensor<T>),
    Add(AddExpression<T>),
}

pub struct Expression<T> {
    variant: Variant<T>,
    output: Option<Tensor<T>>,
    probe: Option<Probe<T>>,
}

impl<T> Expression<T> {
    pub fn new(variant: Variant<T>) -> Self {
        Self {
            variant,
            output: None,
            probe: None,
        }
    }

    pub fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut inputs = Vec::new();
        self.collect_inputs(&mut inputs);
        inputs
    }

    pub fn shape(&self) -> &[usize] {
        match &self.variant {
            Variant::Tensor(tensor) => tensor.shape(),
            Variant::Add(add) => add.shape(),
        }
    }

    pub fn count(&self) -> usize {
        match &self.variant {
            Variant::Tensor(tensor) => tensor.count(),
            Variant::Add(add) => add.count(),
        }
    }

    pub fn output(&mut self, tengu: Arc<Tengu>) -> &Tensor<T> {
        let output = tengu.tensor(self.shape()).empty();
        self.output = Some(output);
        self.output.as_ref().expect("Should have output after creating one")
    }

    fn collect_inputs<'a>(&'a self, inputs: &mut Vec<&'a Tensor<T>>) {
        match &self.variant {
            Variant::Tensor(tensor) => inputs.push(tensor),
            Variant::Add(add) => add.collect_inputs(inputs),
        }
    }
}

// Constructors

impl<T> Expression<T> {
    pub fn tensor(tensor: Tensor<T>) -> Self {
        Self::new(Variant::Tensor(tensor))
    }

    pub fn add(lhs: Tensor<T>, rhs: Tensor<T>) -> Self {
        let lhs = Self::tensor(lhs);
        let rhs = Self::tensor(rhs);
        let add_expression = AddExpression::new(lhs, rhs);
        Self::new(Variant::Add(add_expression))
    }
}

// Trait implementations

impl<T: 'static> Emit for Expression<T> {
    fn emit(&self) -> String {
        match &self.variant {
            Variant::Tensor(tensor) => tensor.emit(),
            Variant::Add(add) => add.emit(),
        }
    }
}

impl_probable!(Expression<T>);
