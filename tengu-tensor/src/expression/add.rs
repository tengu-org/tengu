use crate::{Computation, Tensor};

use super::Expression;

pub struct AddExpression<T> {
    lhs: Box<Expression<T>>,
    rhs: Box<Expression<T>>,
}

impl<T> AddExpression<T> {
    pub fn new(lhs: Expression<T>, rhs: Expression<T>) -> Self {
        Self {
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }

    pub fn count(&self) -> usize {
        self.lhs.count().max(self.rhs.count())
    }

    pub fn shape(&self) -> &[usize] {
        if self.lhs.count() > self.rhs.count() {
            self.lhs.shape()
        } else {
            self.rhs.shape()
        }
    }

    pub fn collect_inputs<'a>(&'a self, inputs: &mut Vec<&'a Tensor<T>>) {
        self.lhs.collect_inputs(inputs);
        self.rhs.collect_inputs(inputs);
    }
}

impl<'a, T> Computation for AddExpression<T> {
    fn emit(&self, idx: &str) -> String {
        let lhs = self.lhs.emit(idx);
        let rhs = self.rhs.emit(idx);
        format!("({lhs}[{idx}] + {rhs}[{idx}])")
    }
}
