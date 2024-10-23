use crate::tensor::Tensor;

use super::{AsSource, Borrowed, Owned, Source};

macro_rules! impl_cast {
    ( $fn:ident, $type:ty ) => {
        pub fn $fn(&self) -> Self {
            match self {
                Source::Owned(owned) => match owned {
                    Owned::Bool(tensor) => <Tensor<$type> as AsSource>::into_source(Tensor::<$type>::from(tensor)),
                    _ => panic!("sdfg"),
                },
                Source::Borrowed(borrowed) => match borrowed {
                    Borrowed::Bool(tensor) => <Tensor<$type> as AsSource>::into_source(Tensor::<$type>::from(*tensor)),
                    _ => panic!("sdfg"),
                },
            }
        }
    };
}

impl<'a> Source<'a> {
    impl_cast!(cast_bool, bool);
    impl_cast!(cast_u32, u32);
    impl_cast!(cast_i32, i32);
    impl_cast!(cast_f32, f32);
}
