use std::ops::{Add, Div, Mul, Sub};

use super::{binary::Binary, Expression};
use crate::WGSLType;

// Add

impl<T: WGSLType> Add for Expression<T> {
    type Output = Expression<T>;

    fn add(self, rhs: Expression<T>) -> Self::Output {
        assert!(self.match_shape(&rhs), "tensor shapes should match");
        Binary::add(self, rhs)
    }
}

impl<T: WGSLType> Add<T> for Expression<T> {
    type Output = Expression<T>;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = Expression::Scalar(rhs);
        Binary::add(self, rhs)
    }
}

// Subtract

impl<T: WGSLType> Sub for Expression<T> {
    type Output = Expression<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        assert!(self.match_shape(&rhs), "tensor shapes should match");
        Binary::sub(self, rhs)
    }
}

impl<T: WGSLType> Sub<T> for Expression<T> {
    type Output = Expression<T>;

    fn sub(self, rhs: T) -> Self::Output {
        let rhs = Expression::Scalar(rhs);
        Binary::sub(self, rhs)
    }
}

// Multiply

impl<T: WGSLType> Mul for Expression<T> {
    type Output = Expression<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        assert!(self.match_shape(&rhs), "tensor shapes should match");
        Binary::mul(self, rhs)
    }
}

impl<T: WGSLType> Mul<T> for Expression<T> {
    type Output = Expression<T>;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs = Expression::Scalar(rhs);
        Binary::mul(self, rhs)
    }
}

// Divide

impl<T: WGSLType> Div for Expression<T> {
    type Output = Expression<T>;

    fn div(self, rhs: Self) -> Self::Output {
        assert!(self.match_shape(&rhs), "tensor shapes should match");
        Binary::div(self, rhs)
    }
}

impl<T: WGSLType> Div<T> for Expression<T> {
    type Output = Expression<T>;

    fn div(self, rhs: T) -> Self::Output {
        let rhs = Expression::Scalar(rhs);
        Binary::div(self, rhs)
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
        let _ = tengu.scalar(2) * lhs.clone() * 3;
        let _ = tengu.scalar(2) / lhs.clone() / 3;
    }

    #[tokio::test]
    async fn tensor_arithmetic() {
        let tengu = Tengu::new().await.unwrap();
        let lhs = tengu.tensor([1, 2, 3]).empty::<i32>();
        let rhs = tengu.tensor([1, 2, 3]).empty::<i32>();
        let _ = lhs.clone() + rhs.clone();
        let _ = lhs.clone() - rhs.clone();
        let _ = lhs.clone() * rhs.clone();
        let _ = lhs.clone() / rhs.clone();
    }

    #[tokio::test]
    #[should_panic]
    async fn shape_mismatch() {
        let tengu = Tengu::new().await.unwrap();
        let lhs = tengu.tensor([1, 2, 3]).empty::<i32>();
        let rhs = tengu.tensor([3, 2, 1]).empty::<i32>();
        let _ = lhs.clone() + rhs.clone();
        let _ = lhs.clone() - rhs.clone();
        let _ = lhs.clone() * rhs.clone();
        let _ = lhs.clone() / rhs.clone();
    }
}
