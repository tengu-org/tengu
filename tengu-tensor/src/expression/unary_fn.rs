use std::{fmt::Display, marker::PhantomData};

use super::traits::{Emit, Node, Shape};
use super::Expression;
use crate::visitor::Visitor;

// Function

#[derive(Copy, Clone)]
pub enum Function {
    Log,
    Exp,
}

impl Function {
    fn symbol(&self) -> &'static str {
        match self {
            Self::Log => "log",
            Self::Exp => "exp",
        }
    }
}

pub struct UnaryFn<T> {
    function: Function,
    expr: Box<dyn Node>,
    phantom: PhantomData<T>,
}

impl<T> UnaryFn<T> {
    pub fn new<S: Clone + Display + 'static>(function: Function, expr: Expression<S>) -> Self {
        Self {
            function,
            expr: Box::new(expr),
            phantom: PhantomData,
        }
    }
}

impl<T> Shape for UnaryFn<T> {
    fn count(&self) -> usize {
        self.expr.count()
    }

    fn shape(&self) -> &[usize] {
        self.expr.shape()
    }
}

impl<T: Display> Emit for UnaryFn<T> {
    fn emit(&self) -> String {
        let expr = self.expr.emit();
        let symbol = self.function.symbol();
        format!("{symbol}({expr})")
    }
}

impl<T: Clone + Display + 'static> Node for UnaryFn<T> {
    fn visit<'a>(&'a self, visitor: &mut Visitor<'a>) {
        self.expr.visit(visitor);
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn source(&self) -> Option<&dyn super::traits::Source> {
        None
    }
}

// Clone

impl<T> Clone for UnaryFn<T> {
    fn clone(&self) -> Self {
        Self {
            function: self.function,
            expr: self.expr.clone_box(),
            phantom: PhantomData,
        }
    }
}
