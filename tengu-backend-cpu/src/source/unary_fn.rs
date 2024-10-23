use tengu_backend_tensor::UnaryFn;

use crate::tensor::Tensor;

use super::{AsSource, Borrowed, Owned, Source};

macro_rules! impl_unary_fn {
    ( $fn:ident, $( [$variant:ident, $type:ty] )+ ) => {
        fn $fn(&self) -> Self {
            match self {
                Source::Owned(owned) => match owned {
                    $(Owned::$variant(tensor) => <Tensor<$type> as AsSource>::into_source(tensor.$fn()),)+
                    other => panic!("{} is not supported for {}", stringify!($fn), other.variant()),
                },
                Source::Borrowed(borrowed) => match borrowed {
                    $(Borrowed::$variant(tensor) => <Tensor<$type> as AsSource>::into_source(tensor.$fn()),)+
                    other => panic!("{} is not supported for {}", stringify!($fn), other.variant()),
                },
            }
        }
    };
}

impl<'a> UnaryFn for Source<'a> {
    impl_unary_fn!(exp, [F32, f32]);
    impl_unary_fn!(log, [F32, f32]);
}
