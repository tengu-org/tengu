use tengu_backend::{Backend, Processor, StorageType};

use super::{Expression, Node, Shape};

pub struct Statement<B: Backend> {
    output: Box<dyn Node<B>>,
    expression: Box<dyn Node<B>>,
}

impl<B: Backend + 'static> Statement<B> {
    pub fn new<T: StorageType>(output: Expression<T, B>, expression: Expression<T, B>) -> Self {
        assert_eq!(expression.shape(), output.shape(), "statement shapes don't match");
        Self {
            output: Box::new(output),
            expression: Box::new(expression),
        }
    }
}

impl<B: Backend> Shape for Statement<B> {
    fn count(&self) -> usize {
        self.output.count()
    }

    fn shape(&self) -> &[usize] {
        self.output.shape()
    }
}

impl<B: Backend + 'static> Node<B> for Statement<B> {
    fn clone_box(&self) -> Box<dyn Node<B>> {
        Box::new(self.clone())
    }

    fn find<'a>(&'a self, label: &str) -> Option<&'a dyn super::Source<B>> {
        self.output.find(label).or_else(|| self.expression.find(label))
    }

    fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor>::Repr {
        let output = self.output.visit(processor);
        let expression = self.expression.visit(processor);
        processor.statement(output, expression)
    }
}

impl<B: Backend> Clone for Statement<B> {
    fn clone(&self) -> Self {
        Self {
            expression: self.expression.clone_box(),
            output: self.output.clone_box(),
        }
    }
}
