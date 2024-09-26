pub struct Size(pub usize);

impl Size {
    pub fn of<T>(data: &[T]) -> Size {
        Self(size_of_val(data))
    }
}

pub trait ByteSize {
    fn bytes(self) -> Size;
}

macro_rules! impl_bytes {
    ($type:ty) => {
        impl ByteSize for $type {
            fn bytes(self) -> Size {
                Size(self as usize)
            }
        }
    };
}

impl_bytes!(u8);
impl_bytes!(u16);
impl_bytes!(u32);
impl_bytes!(u64);
impl_bytes!(usize);
