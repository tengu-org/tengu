use std::{fmt::Display, marker::PhantomData};

use super::{Emit, Node, Shape};
use super::{Expression, Source};
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

impl<T: Clone + Display + 'static> UnaryFn<T> {
    pub fn new(function: Function, expr: Expression<T>) -> Self {
        Self {
            function,
            expr: Box::new(expr),
            phantom: PhantomData,
        }
    }

    pub fn exp(expr: Expression<T>) -> Self {
        Self::new(Function::Exp, expr)
    }

    pub fn log(expr: Expression<T>) -> Self {
        Self::new(Function::Log, expr)
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

    fn source(&self) -> Option<&dyn Source> {
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
