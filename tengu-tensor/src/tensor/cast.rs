use crate::StorageType;

pub trait Cast<S: StorageType> {
    type Output;

    fn cast(&self) -> Self::Output;
}
