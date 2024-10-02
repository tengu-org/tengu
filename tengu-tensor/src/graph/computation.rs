use tengu_backend::{Backend, Processor, StorageType};

use crate::expression::{Expression, Node, Shape, Source};

pub struct Computation<B> {
    statement: Box<dyn Node<B>>,
}

impl<B: Backend + 'static> Computation<B> {
    pub fn new<T: StorageType>(out: Expression<T, B>, expr: Expression<T, B>) -> Self {
        let statement = Box::new(Expression::statement(out, expr));
        Self { statement }
    }

    pub fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor>::Repr {
        self.statement.visit(processor)
    }

    pub fn source(&self, label: &str) -> Option<&dyn Source<B>> {
        self.statement.find(label)
    }
}

// Traits

impl<B> Shape for Computation<B> {
    fn shape(&self) -> &[usize] {
        self.statement.shape()
    }

    fn count(&self) -> usize {
        self.statement.count()
    }
}

// Tests

#[cfg(test)]
mod tests {
    use crate::Tengu;

    use super::*;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn computation_builder() {
        let tengu = Tengu::wgpu().await.unwrap();
        let a = tengu.tensor([2, 2]).init(&[1.0, 2.0, 3.0, 4.0]);
        let b = tengu.tensor([2, 2]).init(&[5.0, 6.0, 7.0, 8.0]);
        let c = tengu.tensor([2, 2]).zero();
        let computation = Computation::new(c, a + b);
        assert_eq!(computation.count(), 4);
    }
}
