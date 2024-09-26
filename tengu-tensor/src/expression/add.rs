use crate::{Emit, Tensor};

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

    pub(crate) fn collect_inputs<'a>(&'a self, inputs: &mut Vec<&'a Tensor<T>>) {
        self.lhs.collect_inputs(inputs);
        self.rhs.collect_inputs(inputs);
    }
}

impl<T: 'static> Emit for AddExpression<T> {
    fn emit(&self) -> String {
        let lhs = self.lhs.emit();
        let rhs = self.rhs.emit();
        format!("({lhs}[idx] + {rhs}[idx])")
    }
}
