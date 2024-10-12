//! Module for unifying the dimensions of tensors.
//!
//! This module provides the `Unify` trait which includes methods for obtaining dimensions and
//! unifying two sets of dimensions into one. It contains an implementation of the trait for
//! slices of `usize` values, representing the dimensions of tensors.
//!
//! The module also includes unit tests to verify the correctness of the unification logic.

use itertools::{EitherOrBoth, Itertools};

/// Trait for unifying dimensions of tensors.
///
/// This trait defines methods for obtaining dimensions and unifying two sets of dimensions into one.
pub trait Unify {
    /// The output type after unification.
    type Output;

    /// Get the dimensions of the tensor.
    ///
    /// # Returns
    /// A slice representing the dimensions of the tensor.
    fn dimensions(&self) -> &[usize];

    /// Unify the dimensions with another tensor.
    ///
    /// # Parameters
    /// - `other`: Another tensor with dimensions to unify.
    ///
    /// # Returns
    /// An `Option` containing the unified dimensions if unification is possible, otherwise `None`.
    ///
    /// # Algorithm
    /// The unification algorithm works by iterating through the dimensions of both tensors from
    /// the last dimension to the first. It checks the dimensions pairwise and applies the following rules:
    /// - If a dimension in one tensor is 1, it takes the corresponding dimension from the other tensor.
    /// - If both dimensions are equal, it takes that dimension.
    /// - If neither of the above conditions is met, unification fails.
    ///
    /// If the tensors have a different number of dimensions, it continues processing until all dimensions are handled,
    /// taking dimensions from the longer tensor where necessary. The process results in a unified shape if successful.
    fn unify(self, other: Self) -> Option<Self::Output>
    where
        Self: Sized;
}

impl Unify for &[usize] {
    type Output = Vec<usize>;

    /// Get the dimensions of the tensor.
    ///
    /// # Returns
    /// A slice representing the dimensions of the tensor.
    fn dimensions(&self) -> &[usize] {
        self
    }

    /// Unify the dimensions with another tensor.
    ///
    /// # Parameters
    /// - `other`: Another tensor with dimensions to unify.
    ///
    /// # Returns
    /// An `Option` containing the unified dimensions if unification is possible, otherwise `None`.
    fn unify(self, other: Self) -> Option<Self::Output> {
        let shape = self
            .dimensions()
            .iter()
            .rev()
            .zip_longest(other.dimensions().iter().rev())
            .map(|v| match v {
                EitherOrBoth::Both(v1, v2) if *v1 == 1 => Some(*v2),
                EitherOrBoth::Both(v1, v2) if *v2 == 1 => Some(*v1),
                EitherOrBoth::Both(v1, v2) if v1 == v2 => Some(*v1),
                EitherOrBoth::Both(_, _) => None,
                EitherOrBoth::Left(v1) => Some(*v1),
                EitherOrBoth::Right(v2) => Some(*v2),
            })
            .rev()
            .collect::<Option<Vec<_>>>()?;
        Some(shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn unify_same_length_fails() {
        let a = &[1, 2, 3, 6];
        let b = &[3, 3, 3, 6];
        a.unify(b).unwrap();
    }

    #[test]
    #[should_panic]
    fn unify_different_lengths_fails() {
        let a = &[2, 4, 3, 6];
        let b = &[3, 1, 1];
        a.unify(b).unwrap();
    }

    #[test]
    fn unify_same_length() {
        let a = &[1, 2, 3, 6];
        let b = &[3, 1, 3, 1];
        let c = a.unify(b).unwrap();
        assert_eq!(c, vec![3, 2, 3, 6]);
    }

    #[test]
    fn unify_different_lengths() {
        let a = &[2, 4, 1, 6];
        let b = &[1, 3, 1];
        let c = a.unify(b).unwrap();
        assert_eq!(c, vec![2, 4, 3, 6]);
    }
}
