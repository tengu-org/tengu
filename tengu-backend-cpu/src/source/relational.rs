use crate::tensor::Tensor;

use super::{AsSource, Borrowed, Owned, Source};

pub trait Equality<Rhs = Self> {
    type Output;

    fn eq(&self, other: &Rhs) -> Self::Output;

    fn neq(&self, other: &Rhs) -> Self::Output;
}

impl<'a> Equality for Source<'a> {
    type Output = Self;

    fn eq(&self, other: &Self) -> Source<'a> {
        match (self, other) {
            (Source::Owned(lhs), Source::Owned(rhs)) => lhs.eq(rhs),
            (Source::Owned(lhs), Source::Borrowed(rhs)) => lhs.eq(rhs),
            (Source::Borrowed(lhs), Source::Owned(rhs)) => lhs.eq(rhs),
            (Source::Borrowed(lhs), Source::Borrowed(rhs)) => lhs.eq(rhs),
        }
    }

    fn neq(&self, other: &Self) -> Source<'a> {
        match (self, other) {
            (Source::Owned(lhs), Source::Owned(rhs)) => lhs.neq(rhs),
            (Source::Owned(lhs), Source::Borrowed(rhs)) => lhs.neq(rhs),
            (Source::Borrowed(lhs), Source::Owned(rhs)) => lhs.neq(rhs),
            (Source::Borrowed(lhs), Source::Borrowed(rhs)) => lhs.neq(rhs),
        }
    }
}

macro_rules! impl_cmp {
    ( $method:ident, $lhs:ident, Borrowed ) => {
        fn $method(&self, other: &Borrowed) -> Self::Output {
            match (self, other) {
                ($lhs::Bool(lhs), Borrowed::Bool(rhs)) => <Tensor<bool> as AsSource>::into_source(lhs.$method(rhs)),
                ($lhs::U32(lhs), Borrowed::U32(rhs)) => <Tensor<bool> as AsSource>::into_source(lhs.$method(rhs)),
                ($lhs::I32(lhs), Borrowed::I32(rhs)) => <Tensor<bool> as AsSource>::into_source(lhs.$method(rhs)),
                ($lhs::F32(lhs), Borrowed::F32(rhs)) => <Tensor<bool> as AsSource>::into_source(lhs.$method(rhs)),
                (lhs, rhs) => panic!(
                    "Comparison operations are not implemented for {} and {}",
                    lhs.variant(),
                    rhs.variant()
                ),
            }
        }
    };
    ( $method:ident, $lhs:ident, Owned ) => {
        fn $method(&self, other: &Owned) -> Self::Output {
            match (self, other) {
                ($lhs::Bool(lhs), Owned::Bool(rhs)) => <Tensor<bool> as AsSource>::into_source(lhs.$method(&rhs)),
                ($lhs::U32(lhs), Owned::U32(rhs)) => <Tensor<bool> as AsSource>::into_source(lhs.$method(&rhs)),
                ($lhs::I32(lhs), Owned::I32(rhs)) => <Tensor<bool> as AsSource>::into_source(lhs.$method(&rhs)),
                ($lhs::F32(lhs), Owned::F32(rhs)) => <Tensor<bool> as AsSource>::into_source(lhs.$method(&rhs)),
                (lhs, rhs) => panic!(
                    "Comparison operations are not implemented for {} and {}",
                    lhs.variant(),
                    rhs.variant()
                ),
            }
        }
    };
}

impl<'a> Equality for Borrowed<'a> {
    type Output = Source<'a>;

    impl_cmp!(eq, Borrowed, Borrowed);
    impl_cmp!(neq, Borrowed, Borrowed);
}

impl<'a> Equality<Owned> for Borrowed<'a> {
    type Output = Source<'a>;

    impl_cmp!(eq, Borrowed, Owned);
    impl_cmp!(neq, Borrowed, Owned);
}

impl Equality for Owned {
    type Output = Source<'static>;

    impl_cmp!(eq, Owned, Owned);
    impl_cmp!(neq, Owned, Owned);
}

impl<'a> Equality<Borrowed<'a>> for Owned {
    type Output = Source<'a>;

    impl_cmp!(eq, Owned, Borrowed);
    impl_cmp!(neq, Owned, Borrowed);
}
