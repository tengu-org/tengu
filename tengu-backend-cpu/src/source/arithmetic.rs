use std::ops::{Add, Div, Mul, Sub};

use super::Source;

macro_rules! impl_op_source {
    ( $op:ident, $trait:ident ) => {
        impl<'a> $trait for &Source<'a> {
            type Output = Source<'a>;

            fn $op(self, rhs: Self) -> Self::Output {
                match (self, rhs) {
                    (Source::U32(_), Source::U32(_)) => (self.as_ref::<u32>().$op(rhs.as_ref::<u32>())).into(),
                    (Source::I32(_), Source::I32(_)) => (self.as_ref::<i32>().$op(rhs.as_ref::<i32>())).into(),
                    (Source::F32(_), Source::F32(_)) => (self.as_ref::<f32>().$op(rhs.as_ref::<f32>())).into(),
                    (lhs, rhs) => panic!(
                        "{} operation not implemented for {} and {}",
                        stringify!($op),
                        lhs.variant(),
                        rhs.variant()
                    ),
                }
            }
        }
    };
}

impl_op_source!(add, Add);
impl_op_source!(sub, Sub);
impl_op_source!(div, Div);
impl_op_source!(mul, Mul);
