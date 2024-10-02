use tengu_backend::{Backend, StorageType};

use super::{Binary, Expression};

impl<T, B> Expression<T, B>
where
    T: StorageType,
    B: Backend + 'static,
{
    pub fn eq(self, rhs: Self) -> Expression<bool, B> {
        Binary::eq(self, rhs)
    }
}
