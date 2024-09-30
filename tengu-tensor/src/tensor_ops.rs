use std::ops::{Add, Sub};

use crate::{Expression, Tensor};

impl<T> Add for Tensor<T> {
    type Output = Expression<T>;

    fn add(self, rhs: Tensor<T>) -> Self::Output {
        assert_eq!(self.shape(), rhs.shape(), "tensor shapes should match");
        Expression::add(self, rhs)
    }
}

impl<T> Sub for Tensor<T> {
    type Output = Expression<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert_eq!(self.shape(), rhs.shape(), "tensor shapes should match");
        Expression::sub(self, rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::Tengu;

    #[tokio::test]
    #[should_panic]
    async fn test_add_shape_mismatch() {
        let tengu = Tengu::new().await.unwrap();
        let lhs = tengu.tensor([1, 2, 3]).empty::<i32>();
        let rhs = tengu.tensor([3, 2, 1]).empty::<i32>();
        let _ = lhs + rhs;
    }

    #[tokio::test]
    #[should_panic]
    async fn test_sub_shape_mismatch() {
        let tengu = Tengu::new().await.unwrap();
        let lhs = tengu.tensor([1, 2, 3]).empty::<i32>();
        let rhs = tengu.tensor([3, 2, 1]).empty::<i32>();
        let _ = lhs + rhs;
    }
}
