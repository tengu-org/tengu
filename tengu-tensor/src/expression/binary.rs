use tengu_backend::{Backend, Processor, StorageType};

use super::{Expression, Node, Shape};
use crate::unify::Unify;

// Operator

#[derive(Copy, Clone)]
enum Operator {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Neq,
}

impl Operator {
    fn symbol(&self) -> &str {
        match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Eq => "==",
            Self::Neq => "!=",
        }
    }
}

// Binary expression

pub struct Binary<B> {
    operator: Operator,
    shape: Vec<usize>,
    count: usize,
    lhs: Box<dyn Node<B>>,
    rhs: Box<dyn Node<B>>,
}

impl<B: Backend + 'static> Binary<B> {
    fn new<T: StorageType>(operator: Operator, lhs: Expression<T, B>, rhs: Expression<T, B>) -> Self {
        let shape = lhs.shape().unify(rhs.shape()).expect("Shapes don't match");
        let count = shape.iter().product();
        Self {
            operator,
            shape,
            count,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

impl<B> Shape for Binary<B> {
    fn count(&self) -> usize {
        self.count
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<B> Node<B> for Binary<B>
where
    B: Backend + 'static,
{
    fn clone_box(&self) -> Box<dyn Node<B>> {
        Box::new(self.clone())
    }

    fn find<'a>(&'a self, label: &str) -> Option<&'a dyn super::Source<B>> {
        self.lhs.find(label).or_else(|| self.rhs.find(label))
    }

    fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor>::Repr {
        let lhs = self.lhs.visit(processor);
        let rhs = self.rhs.visit(processor);
        processor.binary(lhs, rhs, self.operator.symbol())
    }
}

impl<B: Backend> Clone for Binary<B> {
    fn clone(&self) -> Self {
        Self {
            operator: self.operator,
            shape: self.shape.clone(),
            count: self.count,
            lhs: self.lhs.clone_box(),
            rhs: self.rhs.clone_box(),
        }
    }
}

// Constructors

impl<B: Backend + 'static> Binary<B> {
    pub fn add<T: StorageType>(lhs: Expression<T, B>, rhs: Expression<T, B>) -> Expression<T, B> {
        Expression::Binary(Self::new(Operator::Add, lhs, rhs))
    }

    pub fn sub<T: StorageType>(lhs: Expression<T, B>, rhs: Expression<T, B>) -> Expression<T, B> {
        Expression::Binary(Self::new(Operator::Sub, lhs, rhs))
    }

    pub fn mul<T: StorageType>(lhs: Expression<T, B>, rhs: Expression<T, B>) -> Expression<T, B> {
        Expression::Binary(Self::new(Operator::Mul, lhs, rhs))
    }

    pub fn div<T: StorageType>(lhs: Expression<T, B>, rhs: Expression<T, B>) -> Expression<T, B> {
        Expression::Binary(Self::new(Operator::Div, lhs, rhs))
    }

    pub fn eq<T: StorageType>(lhs: Expression<T, B>, rhs: Expression<T, B>) -> Expression<bool, B> {
        Expression::Binary(Binary::new(Operator::Eq, lhs, rhs))
    }

    pub fn neq<T: StorageType>(lhs: Expression<T, B>, rhs: Expression<T, B>) -> Expression<bool, B> {
        Expression::Binary(Binary::new(Operator::Neq, lhs, rhs))
    }
}

#[cfg(test)]
mod tests {
    use crate::expression::Shape;
    use crate::Tengu;

    #[tokio::test]
    async fn propagation() {
        let tengu = Tengu::wgpu().await.unwrap();
        let lhs = tengu.tensor([4, 1, 3]).zero::<f32>();
        let rhs = tengu.tensor([2, 3]).zero::<f32>();
        let add = lhs + rhs;
        assert_eq!(add.shape(), &[4, 2, 3]);
        assert_eq!(add.count(), 24);
    }
}
