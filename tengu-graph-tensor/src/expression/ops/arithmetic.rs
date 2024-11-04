//! This module defines arithmetic operations for tensor expressions, including addition, subtraction,
//! multiplication, and division. It leverages Rust's operator overloading to provide intuitive syntax
//! for tensor arithmetic and includes tests to ensure correct behavior.

use std::ops::{Add, Div, Mul, Sub};
use tengu_backend::Backend;
use tengu_tensor::StorageType;

use super::super::{Arithmetic, Expression};

/// A macro to implement arithmetic operations for tensor expressions using Rust's operator overloading.
///
/// This macro generates implementations for the specified trait and method, enabling the use of standard
/// arithmetic operators (`+`, `-`, `*`, `/`) with `Expression` types. It supports two forms of operations:
/// - Between two `Expression` instances.
/// - Between an `Expression` and a scalar value.
///
/// # Parameters
/// - `$trait`: The trait to implement (e.g., `Add`).
/// - `$method`: The method corresponding to the trait (e.g., `add`).
macro_rules! impl_op {
    ( $trait:ident, $method:ident ) => {
        impl<T, S1, S2, B> $trait<Expression<T, S2, B>> for Expression<T, S1, B>
        where
            T: StorageType,
            S1: StorageType,
            S2: StorageType,
            B: Backend + 'static,
        {
            type Output = Expression<T, T, B>;

            fn $method(self, rhs: Expression<T, S2, B>) -> Self::Output {
                Arithmetic::$method(self, rhs)
            }
        }

        impl<T, S, B> $trait<T> for Expression<T, S, B>
        where
            T: StorageType,
            S: StorageType,
            B: Backend + 'static,
        {
            type Output = Expression<T, T, B>;

            fn $method(self, rhs: T) -> Self::Output {
                let rhs: Expression<T, T, B> = Expression::Scalar(rhs);
                Arithmetic::$method(self, rhs)
            }
        }
    };
}

impl_op!(Add, add);
impl_op!(Sub, sub);
impl_op!(Mul, mul);
impl_op!(Div, div);
