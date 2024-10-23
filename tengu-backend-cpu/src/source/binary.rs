use std::ops::{Add, Div, Mul, Sub};

use crate::tensor::Tensor;

use super::{AsSource, Borrowed, Owned, Source};

// NOTE: Operation trait implementation for Source.

macro_rules! impl_op_source {
    ( $op:ident, $trait:ident ) => {
        impl<'a> $trait for &Source<'a> {
            type Output = Source<'a>;

            fn $op(self, rhs: Self) -> Self::Output {
                match (self, rhs) {
                    (Source::Owned(lhs), Source::Owned(rhs)) => lhs + rhs,
                    (Source::Owned(lhs), Source::Borrowed(rhs)) => lhs + rhs,
                    (Source::Borrowed(lhs), Source::Owned(rhs)) => lhs + rhs,
                    (Source::Borrowed(lhs), Source::Borrowed(rhs)) => lhs + rhs,
                }
            }
        }
    };
}

impl_op_source!(add, Add);
impl_op_source!(sub, Sub);
impl_op_source!(div, Div);
impl_op_source!(mul, Mul);

// NOTE: Operation traits implementation for Owned and Borrowed.

macro_rules! impl_op {
    ( $op:ident, $lhs:ident, $rhs:ident ) => {
        fn $op(self, rhs: &$rhs) -> Self::Output {
            match (self, rhs) {
                ($lhs::U32(lhs), $rhs::U32(rhs)) => <Tensor<u32> as AsSource>::into_source(lhs.add(&rhs)),
                ($lhs::I32(lhs), $rhs::I32(rhs)) => <Tensor<i32> as AsSource>::into_source(lhs.add(&rhs)),
                ($lhs::F32(lhs), $rhs::F32(rhs)) => <Tensor<f32> as AsSource>::into_source(lhs.add(&rhs)),
                (lhs, rhs) => panic!(
                    "{} operation not implemented for {} and {}",
                    stringify!($op),
                    lhs.variant(),
                    rhs.variant()
                ),
            }
        }
    };
}

macro_rules! impl_op_trait {
    ( $op:ident, $trait:ident ) => {
        impl<'a> $trait for &Borrowed<'a> {
            type Output = Source<'a>;

            impl_op!($op, Borrowed, Borrowed);
        }

        impl<'a> $trait<&Owned> for &Borrowed<'a> {
            type Output = Source<'a>;

            impl_op!($op, Borrowed, Owned);
        }

        impl $trait for &Owned {
            type Output = Source<'static>;

            impl_op!($op, Owned, Owned);
        }

        impl<'a> $trait<&Borrowed<'a>> for &Owned {
            type Output = Source<'a>;

            impl_op!($op, Owned, Borrowed);
        }
    };
}

impl_op_trait!(add, Add);
impl_op_trait!(sub, Sub);
impl_op_trait!(mul, Mul);
impl_op_trait!(div, Div);

// NOTE: Equality implementation.

macro_rules! impl_cmp_borrowed {
    ( $lhs:ident ) => {
        fn eq(&self, other: &Borrowed) -> bool {
            match (self, other) {
                ($lhs::Bool(lhs), Borrowed::Bool(rhs)) => lhs.eq(rhs),
                ($lhs::U32(lhs), Borrowed::U32(rhs)) => lhs.eq(rhs),
                ($lhs::I32(lhs), Borrowed::I32(rhs)) => lhs.eq(rhs),
                ($lhs::F32(lhs), Borrowed::F32(rhs)) => lhs.eq(rhs),
                (lhs, rhs) => panic!(
                    "Comparison operations are not implemented for {} and {}",
                    lhs.variant(),
                    rhs.variant()
                ),
            }
        }
    };
}

macro_rules! impl_cmp_owned {
    ( $lhs:ident ) => {
        fn eq(&self, other: &Owned) -> bool {
            match (self, other) {
                ($lhs::Bool(lhs), Owned::Bool(rhs)) => lhs.eq(&rhs),
                ($lhs::U32(lhs), Owned::U32(rhs)) => lhs.eq(&rhs),
                ($lhs::I32(lhs), Owned::I32(rhs)) => lhs.eq(&rhs),
                ($lhs::F32(lhs), Owned::F32(rhs)) => lhs.eq(&rhs),
                (lhs, rhs) => panic!(
                    "Comparison operations are not implemented for {} and {}",
                    lhs.variant(),
                    rhs.variant()
                ),
            }
        }
    };
}

impl<'a> PartialEq for Borrowed<'a> {
    impl_cmp_borrowed!(Borrowed);
}

impl<'a> PartialEq<Owned> for Borrowed<'a> {
    impl_cmp_owned!(Borrowed);
}

impl PartialEq for Owned {
    impl_cmp_owned!(Owned);
}

impl<'a> PartialEq<Borrowed<'a>> for Owned {
    impl_cmp_borrowed!(Owned);
}
