use super::Binary;
use crate::{expression::Expression, WGSLType};

impl<T: WGSLType> Expression<T> {
    pub fn eq(self, rhs: Self) -> Expression<bool> {
        assert!(self.unify(&rhs).is_some(), "tensor shapes should match");
        Binary::eq(self, rhs)
    }
}
