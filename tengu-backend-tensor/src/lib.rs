//! This crate provides the foundational traits and types that other tensor implementations must
//! adhere to in order to support tensor operations. The crate is intended to be consumed by user-facing
//! crates such as `tengu-graph`.
//!
//! ## Features
//!
//! - GPU Compatibility: Defines traits (IOType and StorageType) that ensure data types used in tensors are compatible
//!   with GPU operations and can be safely transferred between CPU and GPU memory.
//! - Tensor Abstraction: Provides the Tensor trait, representing a tensor with essential methods for managing and retrieving
//!   tensor data asynchronously.
//! - Type Flexibility: Supports various types (f32, u32, i32) for tensor elements, enabling efficient computation and data management.
//!
//! ## Modules
//!
//! - `tensor`: Defines the `Tensor` trait, which represents a tensor in the computation framework.
//!
//! - `types`: Defines traits used throughout the crate, namely `IOType` and `StorageType`,
//!   which represent different categories of data types used in tensor computations.
//!
//! ## Traits
//! - IOType: Represents types transferable between CPU and GPU, ensuring safe memory management.
//! - StorageType: Specifies types that can be stored on the GPU, with an associated IOType for CPU transfers.
//! - Tensor: Defines the core interface for tensors, including methods like:
//!    - label: Returns the tensor's label.
//!    - count: Provides the number of elements.
//!    - shape: Retrieves the tensor's shape.
//!    - retrieve: Asynchronously accesses tensor data.

mod cast;
mod function;
mod operator;
mod tensor;
mod types;
mod utils;

pub use cast::Type;
pub use function::{Function, UnaryFn};
pub use operator::Operator;
pub use tensor::Tensor;
pub use types::{IOType, StorageType};
pub use utils::*;
