use std::ops::{Add, Sub};

use crate::{
    expression::{AddExpression, SubExpression},
    Expression, WGSLType,
};

// Add

impl<T: WGSLType> Add for Expression<T> {
    type Output = Expression<T>;

    fn add(self, rhs: Expression<T>) -> Self::Output {
        assert_eq!(self.shape(), rhs.shape(), "tensor shapes should match");
        Expression::Add(AddExpression::new(self, rhs))
    }
}

impl<T: WGSLType> Add<T> for Expression<T> {
    type Output = Expression<T>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = Expression::Scalar(rhs);
        Expression::Add(AddExpression::new(self, rhs))
    }
}

// Subtract

impl<T: WGSLType> Sub for Expression<T> {
    type Output = Expression<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape(), rhs.shape(), "tensor shapes should match");
        Expression::Sub(SubExpression::new(self, rhs))
    }
}

impl<T: WGSLType> Sub<T> for Expression<T> {
    type Output = Expression<T>;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs = Expression::Scalar(rhs);
        Expression::Sub(SubExpression::new(self, rhs))
    }
}

#[cfg(test)]
mod tests {
    use crate::Tengu;

    #[tokio::test]
    async fn tensor_scalar_arithmetic() {
        let tengu = Tengu::new().await.unwrap();
        let lhs = tengu.tensor([1, 2, 3]).empty::<i32>();
        let _ = tengu.scalar(1) + lhs.clone() + 2;
        let _ = tengu.scalar(1) - lhs.clone() - 1;
    }

    #[tokio::test]
    async fn tensor_arithmetic() {
        let tengu = Tengu::new().await.unwrap();
        let lhs = tengu.tensor([1, 2, 3]).empty::<i32>();
        let rhs = tengu.tensor([1, 2, 3]).empty::<i32>();
        let _ = lhs.clone() + rhs.clone();
        let _ = lhs.clone() - rhs.clone();
    }

    #[tokio::test]
    #[should_panic]
    async fn shape_mismatch() {
        let tengu = Tengu::new().await.unwrap();
        let lhs = tengu.tensor([1, 2, 3]).empty::<i32>();
        let rhs = tengu.tensor([3, 2, 1]).empty::<i32>();
        let _ = lhs.clone() + rhs.clone();
        let _ = lhs.clone() - rhs.clone();
    }
}
