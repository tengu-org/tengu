//! This module defines relational operations for tensor expressions, specifically the equality operation.
//! It leverages the backend processing capabilities to apply these operations on tensor data.

use tengu_backend::Backend;
use tengu_tensor_traits::StorageType;

use super::{Binary, Expression};

impl<T, B> Expression<T, B>
where
    T: StorageType,
    B: Backend + 'static,
{
    /// Compares two tensor expressions for equality.
    ///
    /// # Parameters
    /// - `rhs`: The right-hand side tensor expression to compare.
    ///
    /// # Returns
    /// An `Expression` of type `bool` indicating the result of the equality comparison.
    pub fn eq(self, rhs: Self) -> Expression<bool, B> {
        Binary::eq(self, rhs)
    }
}
