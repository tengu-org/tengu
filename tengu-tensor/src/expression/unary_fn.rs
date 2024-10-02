use tengu_backend::{Backend, Processor, StorageType};

use super::{Expression, Node, Shape};

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

pub struct UnaryFn<B> {
    function: Function,
    expression: Box<dyn Node<B>>,
}

impl<B: Backend + 'static> UnaryFn<B> {
    pub fn new<T: StorageType>(function: Function, expr: Expression<T, B>) -> Self {
        Self {
            function,
            expression: Box::new(expr),
        }
    }

    pub fn exp<T: StorageType>(expr: Expression<T, B>) -> Self {
        Self::new(Function::Exp, expr)
    }

    pub fn log<T: StorageType>(expr: Expression<T, B>) -> Self {
        Self::new(Function::Log, expr)
    }
}

impl<B> Shape for UnaryFn<B> {
    fn count(&self) -> usize {
        self.expression.count()
    }

    fn shape(&self) -> &[usize] {
        self.expression.shape()
    }
}

impl<B: Backend + 'static> Node<B> for UnaryFn<B> {
    fn clone_box(&self) -> Box<dyn Node<B>> {
        Box::new(self.clone())
    }

    fn find<'a>(&'a self, label: &str) -> Option<&'a dyn super::Source<B>> {
        self.expression.find(label)
    }

    fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor>::Repr {
        let expr = self.expression.visit(processor);
        let symbol = self.function.symbol();
        processor.unary_fn(expr, symbol)
    }
}

impl<B: Backend> Clone for UnaryFn<B> {
    fn clone(&self) -> Self {
        Self {
            function: self.function,
            expression: self.expression.clone_box(),
        }
    }
}
