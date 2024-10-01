use std::rc::Rc;

use crate::expression::traits::{Emit, Node, Shape};
use crate::expression::Expression;
use crate::visitor::Visitor;
use crate::{Tengu, WGSLType};

pub struct Computation {
    expression: Box<dyn Node>,
    output: Box<dyn Node>,
}

impl Computation {
    pub fn new<T: WGSLType>(tengu: &Rc<Tengu>, label: impl Into<String>, expression: Expression<T>) -> Self {
        let output = tengu.tensor(expression.shape()).with_label(label).empty::<T>();
        Self {
            expression: Box::new(expression),
            output: Box::new(output),
        }
    }
}

// Traits

impl Shape for Computation {
    fn shape(&self) -> &[usize] {
        self.output.shape()
    }

    fn count(&self) -> usize {
        self.output.count()
    }
}

impl Emit for Computation {
    fn emit(&self) -> String {
        format!("{} = {};", self.output.emit(), self.expression.emit())
    }
}

impl Node for Computation {
    fn visit<'a>(&'a self, visitor: &mut Visitor<'a>) {
        self.expression.visit(visitor);
        self.output.visit(visitor);
    }

    fn clone_box(&self) -> Box<dyn Node> {
        panic!("computations should not be exposed and thus should not be cloned")
    }
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[tokio::test]
    async fn computation_builder() {
        let tengu = Tengu::new().await.unwrap();
        let a = tengu.tensor([2, 2]).init(&[1.0, 2.0, 3.0, 4.0]);
        let b = tengu.tensor([2, 2]).init(&[5.0, 6.0, 7.0, 8.0]);
        let computation = Computation::new(&tengu, "c", a + b);
        assert_eq!(computation.count(), 4);
    }

    #[tokio::test]
    async fn computation_emit() {
        let tengu = Tengu::new().await.unwrap();
        let a = tengu.tensor([2, 2]).with_label("a").empty::<f32>();
        let b = tengu.tensor([2, 2]).with_label("b").empty::<f32>();
        let computation = Computation::new(&tengu, "c", a + b);
        assert_eq!(computation.emit(), "c[idx] = (a[idx] + b[idx]);");
    }
}
