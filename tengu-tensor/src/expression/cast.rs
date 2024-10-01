use std::{fmt::Display, marker::PhantomData};

use super::traits::{Emit, Node, Shape};
use super::Expression;
use crate::visitor::Visitor;

pub struct Cast<T> {
    expr: Box<dyn Node>,
    phantom: PhantomData<T>,
}

impl<T> Cast<T> {
    pub fn new<S: Clone + Display + 'static>(expr: Expression<S>) -> Self {
        Self {
            expr: Box::new(expr),
            phantom: PhantomData,
        }
    }
}

impl<T> Shape for Cast<T> {
    fn count(&self) -> usize {
        self.expr.count()
    }

    fn shape(&self) -> &[usize] {
        self.expr.shape()
    }
}

impl<T: Display> Emit for Cast<T> {
    fn emit(&self) -> String {
        let expr = self.expr.emit();
        let type_name = std::any::type_name::<T>();
        format!("{type_name}({expr})")
    }
}

impl<T: Clone + Display + 'static> Node for Cast<T> {
    fn visit<'a>(&'a self, visitor: &mut Visitor<'a>) {
        self.expr.visit(visitor);
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }
}

// Clone

impl<T> Clone for Cast<T> {
    fn clone(&self) -> Self {
        Self {
            expr: self.expr.clone_box(),
            phantom: PhantomData,
        }
    }
}
