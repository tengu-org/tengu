use std::ops::{Add, Div, Mul, Sub};
use tengu_backend::{Backend, StorageType};

use super::{Binary, Expression};

macro_rules! impl_op {
    ( $trait:ident, $method:ident ) => {
        impl<T, B> $trait for Expression<T, B>
        where
            T: StorageType,
            B: Backend + 'static,
        {
            type Output = Expression<T, B>;

            fn $method(self, rhs: Expression<T, B>) -> Self::Output {
                Binary::$method(self, rhs)
            }
        }

        impl<T, B> $trait<T> for Expression<T, B>
        where
            T: StorageType,
            B: Backend + 'static,
        {
            type Output = Expression<T, B>;

            fn $method(self, rhs: T) -> Self::Output {
                let rhs = Expression::Scalar(rhs);
                Binary::$method(self, rhs)
            }
        }
    };
}

impl_op!(Add, add);
impl_op!(Sub, sub);
impl_op!(Mul, mul);
impl_op!(Div, div);

#[cfg(test)]
mod tests {
    use crate::Tengu;

    #[tokio::test]
    async fn tensor_scalar_arithmetic() {
        let tengu = Tengu::wgpu().await.unwrap();
        let lhs = tengu.tensor([1, 2, 3]).zero::<i32>();
        let _ = tengu.scalar(1) + lhs.clone() + 2;
        let _ = tengu.scalar(1) - lhs.clone() - 1;
        let _ = tengu.scalar(2) * lhs.clone() * 3;
        let _ = tengu.scalar(2) / lhs.clone() / 3;
    }

    #[tokio::test]
    async fn tensor_arithmetic() {
        let tengu = Tengu::wgpu().await.unwrap();
        let lhs = tengu.tensor([1, 2, 3]).zero::<i32>();
        let rhs = tengu.tensor([1, 2, 3]).zero::<i32>();
        let _ = lhs.clone() + rhs.clone();
        let _ = lhs.clone() - rhs.clone();
        let _ = lhs.clone() * rhs.clone();
        let _ = lhs.clone() / rhs.clone();
    }

    #[tokio::test]
    #[should_panic]
    async fn shape_mismatch() {
        let tengu = Tengu::wgpu().await.unwrap();
        let lhs = tengu.tensor([2, 3]).zero::<i32>();
        let rhs = tengu.tensor([4]).zero::<i32>();
        let _ = lhs.clone() + rhs.clone();
        let _ = lhs.clone() - rhs.clone();
        let _ = lhs.clone() * rhs.clone();
        let _ = lhs.clone() / rhs.clone();
    }
}
