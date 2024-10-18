//! Tensor abstraction and related types for the Tengu backend.
//!
//! This module defines the `Tensor` trait which represents a tensor in the Tengu backend.
//! Tensors are fundamental data structures used in numerical computations and are typically
//! stored on the GPU. This trait outlines the necessary operations and associated types
//! required for a particular implemntation on some backend.
//!
//! # Traits
//!
//! - `Tensor`: A trait for representing tensors in the Tengu backend.
//!
//! # Related Traits
//!
//! - `StorageType`: A trait representing types that can be stored on the GPU.
//! - `Probe`: A trait representing probes that can be bound to tensors.

use crate::{Probe, StorageType};

/// A trait for representing tensors in the Tengu backend.
///
/// This trait defines the necessary operations and associated types required for
/// interacting with tensors in a generic and type-safe manner. Implementors of
/// this trait must specify the type of probe that can be bound to the tensor and
/// provide methods for retrieving the tensor's label and probe.
///
/// # Associated Types
/// - `Probe`: The type of probe that can be bound to the tensor.
///
/// # Methods
/// - `label`: Returns the label of the tensor.
/// - `probe`: Gets the tensor's probe or initializes a new one and binds it to this tensor.
pub trait Tensor<T: StorageType> {
    /// The type of probe this tensor can be bound to.
    type Probe: Probe<T::IOType>;

    /// Returns the label of the tensor.
    ///
    /// # Returns
    /// A string slice representing the label of the tensor.
    fn label(&self) -> &str;

    /// Gets the tensor's probe or initializes a new one and binds it to this tensor.
    ///
    /// # Returns
    /// A reference to the probe bound to this tensor.
    fn probe(&self) -> Self::Probe;
}
