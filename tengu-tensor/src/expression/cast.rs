use std::marker::PhantomData;

use tengu_backend::{Backend, Processor, StorageType};

use super::{Expression, Node, Shape};

pub struct Cast<T, B> {
    expression: Box<dyn Node<B>>,
    phantom: PhantomData<T>,
}

impl<T: StorageType, B: Backend + 'static> Cast<T, B> {
    pub fn new<S: StorageType>(expr: Expression<S, B>) -> Self {
        Self {
            expression: Box::new(expr),
            phantom: PhantomData,
        }
    }
}

impl<T, B> Shape for Cast<T, B> {
    fn count(&self) -> usize {
        self.expression.count()
    }

    fn shape(&self) -> &[usize] {
        self.expression.shape()
    }
}

impl<T, B> Node<B> for Cast<T, B>
where
    T: Clone + 'static,
    B: Backend + 'static,
{
    fn clone_box(&self) -> Box<dyn Node<B>> {
        Box::new(self.clone())
    }

    fn find<'a>(&'a self, label: &str) -> Option<&'a dyn super::Source<B>> {
        self.expression.find(label)
    }

    fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor>::Repr {
        let expr = self.expression.visit(processor);
        let ty = std::any::type_name::<T>().to_string();
        processor.cast(expr, &ty)
    }
}

impl<T, B: Backend> Clone for Cast<T, B> {
    fn clone(&self) -> Self {
        Self {
            expression: self.expression.clone_box(),
            phantom: PhantomData,
        }
    }
}
