//! This module defines the `Limits` trait, which is used to give user information about particular
//! backend limitations such as maximum number of tensors in one compute stage.

/// The `Limits` trait defines a set of operations for querying the limitations of a backend.
pub trait Limits {
    type Backend;

    /// Returns the maximum number of tensors that can be used in a single compute stage.
    ///
    /// # Returns
    /// The maximum number of tensors that can be used in a single compute stage.
    fn max_tensor_per_compute(&self) -> Option<usize>;
}
