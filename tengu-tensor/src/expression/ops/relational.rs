use super::Binary;
use crate::{expression::Expression, StorageType};

impl<T: StorageType> Expression<T> {
    pub fn eq(self, rhs: Self) -> Expression<bool> {
        assert!(self.unify(&rhs).is_some(), "tensor shapes should match");
        Binary::eq(self, rhs)
    }
}
