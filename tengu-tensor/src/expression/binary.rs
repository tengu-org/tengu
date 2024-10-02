use std::{fmt::Display, marker::PhantomData};

use super::traits::{Emit, Node, Shape};
use super::Expression;
use crate::unify::Unify;
use crate::visitor::Visitor;

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

pub struct Binary<T> {
    operator: Operator,
    shape: Vec<usize>,
    count: usize,
    lhs: Box<dyn Node>,
    rhs: Box<dyn Node>,
    phantom: PhantomData<T>,
}

impl<T> Binary<T> {
    fn new<S: Clone + Display + 'static>(operator: Operator, lhs: Expression<S>, rhs: Expression<S>) -> Self {
        let shape = lhs.shape().unify(rhs.shape()).expect("Shapes don't match");
        let count = shape.iter().product();
        Self {
            operator,
            shape,
            count,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
            phantom: PhantomData,
        }
    }
}

impl<T> Shape for Binary<T> {
    fn count(&self) -> usize {
        self.count
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl<T: Display> Emit for Binary<T> {
    fn emit(&self) -> String {
        let lhs = self.lhs.emit();
        let rhs = self.rhs.emit();
        let symbol = self.operator.symbol();
        format!("({lhs} {symbol} {rhs})")
    }
}

impl<T: Clone + Display + 'static> Node for Binary<T> {
    fn visit<'a>(&'a self, visitor: &mut Visitor<'a>) {
        self.lhs.visit(visitor);
        self.rhs.visit(visitor);
    }

    fn clone_box(&self) -> Box<dyn Node> {
        Box::new(self.clone())
    }

    fn source(&self) -> Option<&dyn super::traits::Source> {
        None
    }
}

// Clone

impl<T: Clone + 'static> Clone for Binary<T> {
    fn clone(&self) -> Self {
        Self {
            operator: self.operator,
            shape: self.shape.clone(),
            count: self.count,
            lhs: self.lhs.clone_box(),
            rhs: self.rhs.clone_box(),
            phantom: PhantomData,
        }
    }
}

// Constructors

impl<T: Clone + Display + 'static> Binary<T> {
    pub fn add(lhs: Expression<T>, rhs: Expression<T>) -> Expression<T> {
        Expression::Binary(Self::new(Operator::Add, lhs, rhs))
    }

    pub fn sub(lhs: Expression<T>, rhs: Expression<T>) -> Expression<T> {
        Expression::Binary(Self::new(Operator::Sub, lhs, rhs))
    }

    pub fn mul(lhs: Expression<T>, rhs: Expression<T>) -> Expression<T> {
        Expression::Binary(Self::new(Operator::Mul, lhs, rhs))
    }

    pub fn div(lhs: Expression<T>, rhs: Expression<T>) -> Expression<T> {
        Expression::Binary(Self::new(Operator::Div, lhs, rhs))
    }

    pub fn eq(lhs: Expression<T>, rhs: Expression<T>) -> Expression<bool> {
        Expression::Binary(Binary::new(Operator::Eq, lhs, rhs))
    }

    pub fn neq(lhs: Expression<T>, rhs: Expression<T>) -> Expression<bool> {
        Expression::Binary(Binary::new(Operator::Neq, lhs, rhs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tengu;

    #[tokio::test]
    async fn add_expression() {
        let tengu = Tengu::new().await.unwrap();
        let lhs = tengu.tensor([1, 2, 3]).label("tz_lhs").zero::<f32>();
        let rhs = tengu.tensor([1, 2, 3]).label("tz_rhs").zero::<f32>();
        let add = lhs + rhs;
        assert_eq!(add.count(), 6);
        assert_eq!(add.shape(), &[1, 2, 3]);
        assert_eq!(add.emit(), "(tz_lhs[idx] + tz_rhs[idx])");
    }

    #[tokio::test]
    async fn propagation() {
        let tengu = Tengu::new().await.unwrap();
        let lhs = tengu.tensor([4, 1, 3]).zero::<f32>();
        let rhs = tengu.tensor([2, 3]).zero::<f32>();
        let add = lhs + rhs;
        assert_eq!(add.shape(), &[4, 2, 3]);
        assert_eq!(add.count(), 24);
    }
}
