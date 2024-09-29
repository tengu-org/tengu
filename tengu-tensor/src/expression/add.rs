use crate::Tensor;

use super::Expression;

pub struct AddExpression<T> {
    lhs: Box<Expression<T>>,
    rhs: Box<Expression<T>>,
}

impl<T> AddExpression<T> {
    pub fn new(lhs: Expression<T>, rhs: Expression<T>) -> Self {
        Self {
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
        format!("({lhs} + {rhs})")
    }

    pub(crate) fn collect_inputs<'a>(&'a self, inputs: &mut Vec<&'a Tensor<T>>) {
        self.lhs.collect_inputs(inputs);
        self.rhs.collect_inputs(inputs);
    }
}

#[cfg(test)]
mod tests {
    use crate::Tengu;

    use super::*;

    #[tokio::test]
    async fn test_add_expression() {
        let tengu = Tengu::new().await.unwrap();
        let lhs = Expression::tensor(tengu.tensor::<f32>([1, 2, 3]).with_label("tz_lhs").empty());
        let rhs = Expression::tensor(tengu.tensor::<f32>([1, 2, 3]).with_label("tz_rhs").empty());
        let add = AddExpression::new(lhs, rhs);
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
