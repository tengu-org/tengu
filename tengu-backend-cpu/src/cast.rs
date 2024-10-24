pub trait Cast<To> {
    fn cast(self) -> To;
}

impl<T> Cast<T> for T {
    fn cast(self) -> T {
        self
    }
}

macro_rules! impl_convert {
    ( $from:ty, bool ) => {
        impl Cast<bool> for $from {
            fn cast(self) -> bool {
                self != (0 as $from)
            }
        }
    };
    ( bool, $to:ty ) => {
        impl Cast<$to> for bool {
            fn cast(self) -> $to {
                self as u32 as $to
            }
        }
    };
    ( $from:ty, $to:ty ) => {
        impl Cast<$to> for $from {
            fn cast(self) -> $to {
                self as $to
            }
        }
    };
}

impl_convert!(u32, i32);
impl_convert!(u32, f32);
impl_convert!(u32, bool);
impl_convert!(i32, u32);
impl_convert!(i32, f32);
impl_convert!(i32, bool);
impl_convert!(f32, u32);
impl_convert!(f32, i32);
impl_convert!(f32, bool);
impl_convert!(bool, u32);
impl_convert!(bool, i32);
impl_convert!(bool, f32);
