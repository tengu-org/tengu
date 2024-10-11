//! This module provides functionality for calculating the byte size of various types. It defines
//! the `ByteSize` trait and implements it for several primitive types using a macro.

/// The `ByteSize` trait provides a method for calculating the byte size of a given type.
pub trait ByteSize {
    /// Returns the byte size of the given type `T` multiplied by the value of the implementing type.
    ///
    /// # Parameters
    /// - `self`: The value of the implementing type.
    ///
    /// # Returns
    /// The byte size of the type `T` multiplied by the value of the implementing type.
    fn of<T>(self) -> usize;
}

macro_rules! impl_bytes {
    ($type:ty) => {
        impl ByteSize for $type {
            fn of<T>(self) -> usize {
                self as usize * std::mem::size_of::<T>()
            }
        }
    };
}

impl_bytes!(u8);
impl_bytes!(u16);
impl_bytes!(u32);
impl_bytes!(u64);
impl_bytes!(i8);
impl_bytes!(i16);
impl_bytes!(i32);
impl_bytes!(i64);
impl_bytes!(isize);
impl_bytes!(usize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concrete_type_size() {
        assert_eq!(4, 1_u8.of::<f32>());
        assert_eq!(8, 2_u16.of::<f32>());
        assert_eq!(12, 3_u32.of::<f32>());
        assert_eq!(16, 4_u64.of::<f32>());
        assert_eq!(20, 5_usize.of::<f32>());
        assert_eq!(4, 1_i8.of::<f32>());
        assert_eq!(8, 2_i16.of::<f32>());
        assert_eq!(12, 3_i32.of::<f32>());
        assert_eq!(16, 4_i64.of::<f32>());
        assert_eq!(20, 5_isize.of::<f32>());
    }
}
