//! This crate provides the foundational traits and types that other backend implementations must
//! adhere to in order to support tensor operations. The crate is intended to be consumed by user-facing
//! crates such as `tengu-tensor`.
//!
//! ## Modules
//!
//! - `tensor`: Defines the `Tensor` trait, which represents a tensor in the computation framework.
//!
//! - `types`: Defines traits used throughout the crate, namely `IOType` and `StorageType`,
//!   which represent different categories of data types used in tensor computations.
//!

mod tensor;
mod types;

pub use tensor::Tensor;
pub use types::{IOType, StorageType};
