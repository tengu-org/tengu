/// This module defines lossy type casts between various backend-supported types.

/// Trait for type casts.
pub trait Cast<To> {
    /// Casts the value to the target type.
    ///
    /// # Returns
    /// The value casted to the target type.
    fn cast(self) -> To;
}

impl<T> Cast<T> for T {
    /// Casts the value to the same type. Basically an identity method.
    ///
    /// # Returns
    /// The value passed in without any changes.
    fn cast(self) -> T {
        self
    }
}

/// Implementations of type casts between various types. The following conversions are supported:
/// Conversiions to bool are based on the value being non-zero.
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
