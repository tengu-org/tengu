use crate::{Tensor, WGSLType};

use super::Expression;

// Operator

#[derive(Copy, Clone)]
enum Operator {
    Add,
    Sub,
    Mul,
    Div,
}

impl Operator {
    fn symbol(&self) -> char {
        match self {
            Self::Add => '+',
            Self::Sub => '-',
            Self::Mul => '*',
            Self::Div => '/',
        }
    }
}

// Binary expression

#[derive(Clone)]
pub struct Binary<T> {
    operator: Operator,
    lhs: Box<Expression<T>>,
    rhs: Box<Expression<T>>,
}

impl<T: WGSLType> Binary<T> {
    fn new(operator: Operator, lhs: Expression<T>, rhs: Expression<T>) -> Self {
        Self {
            operator,
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

    pub fn emit(&self) -> String {
        let lhs = self.lhs.emit();
        let rhs = self.rhs.emit();
        let symbol = self.operator.symbol();
        format!("({lhs} {symbol} {rhs})")
    }

    pub(crate) fn collect_inputs<'a>(&'a self, inputs: &mut Vec<&'a Tensor<T>>) {
        self.lhs.collect_inputs(inputs);
        self.rhs.collect_inputs(inputs);
    }
}

// Constructors

impl<T: WGSLType> Binary<T> {
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
}

#[cfg(test)]
mod tests {
    use crate::Tengu;

    #[tokio::test]
    async fn add_expression() {
        let tengu = Tengu::new().await.unwrap();
        let lhs = tengu.tensor([1, 2, 3]).with_label("tz_lhs").empty::<f32>();
        let rhs = tengu.tensor([1, 2, 3]).with_label("tz_rhs").empty::<f32>();
        let add = lhs + rhs;
        assert_eq!(add.count(), 6);
        assert_eq!(add.shape(), &[1, 2, 3]);
        let mut inputs = Vec::new();
        add.collect_inputs(&mut inputs);
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].shape(), &[1, 2, 3]);
        assert_eq!(inputs[1].shape(), &[1, 2, 3]);
        assert_eq!(add.emit(), "(tz_lhs[idx] + tz_rhs[idx])");
    }
}
