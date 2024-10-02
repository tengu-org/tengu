use super::{Binary, Expression};
use crate::StorageType;

impl<T: StorageType> Expression<T> {
    pub fn eq(self, rhs: Self) -> Expression<bool> {
        Binary::eq(self, rhs)
    }
}
