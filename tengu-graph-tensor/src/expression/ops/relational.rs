//! This module defines relational operations for tensor expressions, specifically the equality operation.
//! It leverages the backend processing capabilities to apply these operations on tensor data.

use tengu_backend::Backend;
use tengu_tensor::StorageType;

use super::super::{Expression, Relational};

impl<T, S, B> Expression<T, S, B>
where
    T: StorageType,
    S: StorageType,
    B: Backend + 'static,
{
    /// Compares two tensor expressions for equality.
    ///
    /// # Parameters
    /// - `rhs`: The right-hand side tensor expression to compare.
    ///
    /// # Returns
    /// An `Expression` of type `bool` indicating the result of the equality comparison.
    pub fn eq<U: StorageType>(self, rhs: Expression<T, U, B>) -> Expression<bool, T, B> {
        Relational::eq(self, rhs)
    }
}
