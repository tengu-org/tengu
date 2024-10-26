use tengu_backend_tensor::UnaryFn;

use super::Source;

macro_rules! impl_unary_fn {
    ( $fn:ident, $( [$variant:ident, $type:ty] )+ ) => {
        fn $fn(&self) -> Self {
            match self {
                $(Self::$variant(_) => self.as_ref::<$type>().exp().into())+,
                other => panic!("{} is not supported for {}", stringify!($fn), other.variant()),
            }
        }
    };
}

impl<'a> UnaryFn for Source<'a> {
    impl_unary_fn!(exp, [F32, f32]);
    impl_unary_fn!(log, [F32, f32]);
}
