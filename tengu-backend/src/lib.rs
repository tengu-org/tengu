//! This crate defines an interface for implementing tensor operations on various backends. It is designed
//! to be flexible and extensible, allowing for different types of backends such as CPU, Vulkan, or any other
//! backend. This crate provides the foundational traits and types that other backend implementations must
//! adhere to in order to support tensor operations. The crate is intended to be consumed by user-facing
//! crates such as `tengu-tensor`.
//!
//! ## Modules
//!
//! - `backend`: Defines the `Backend` trait, which serves as the main interface for tensor computation backends.
//!   This trait includes methods for creating and managing tensors, processors, compute instances, linkers, and
//!   readouts.
//!
//! - `compute`: Defines the `Compute` trait, which provides an interface for performing computations
//!   using a given processor state within a specified backend.
//!
//! - `error`: Defines error handling types used throughout the crate. It includes the `Error` type
//!   and the `Result` type alias for simplifying error handling.
//!
//! - `linker`: Defines the `Linker` trait, which is used for linking tensor data between different parts
//!   of a computation graph or between different storage locations.
//!
//! - `probe`: Defines the `Probe` trait, which provides an interface for retrieving data asynchronously
//!   from different types that implement the `IOType` trait.
//!
//! - `processor`: Defines the `Processor` trait, which is used for processing the abstract syntax tree (AST)
//!   of tensor expressions in a final tagless style.
//!
//! - `readout`: Defines the `Readout` trait, which provides an interface for reading out tensor data
//!   from a computation graph.
//!
//! - `limits`: Defines the `Limits` struct, which represents the limits of the backend in terms of
//!   max tensor count and other properties.
//!
//! - `tensor`: Defines the `Tensor` trait, which represents a tensor in the computation framework.
//!   It includes methods for manipulating tensor data.
//!
//! - `types`: Defines various types used throughout the crate, including `IOType` and `StorageType`,
//!   which represent different categories of data types used in tensor computations.
//!
//! ## Traits
//!
//! Implementors of this crate will need to implement the following traits to create a functional backend:
//!
//! - **Backend**: The core trait that defines the overall interface for tensor computation backends.
//!   It includes associated types and methods for creating and managing tensors, processors, compute instances,
//!   linkers, and readouts.
//!
//! - **Compute**: Defines how computations are committed and executed using a given processor state within the backend.
//!
//! - **Linker**: Manages the linking and copying of tensor data between different storage locations or parts
//!   of the computation graph.
//!
//! - **Processor**: Handles the processing of the abstract syntax tree (AST) for tensor expressions in a final tagless style.
//!
//! - **Readout**: Provides methods for reading tensor data from the computation graph.
//!
//! - **Tensor**: Represents a tensor and provides methods for manipulating tensor data.
//!
//! - **Probe**: Allows for the asynchronous retrieval of data from tensors or other types that implement the `IOType` trait.

mod backend;
mod compute;
mod error;
mod limits;
mod linker;
mod probe;
mod processor;
mod readout;
mod tensor;
mod types;

pub use backend::Backend;
pub use compute::Compute;
pub use error::{Error, Result};
pub use limits::Limits;
pub use linker::Linker;
pub use probe::Probe;
pub use processor::Processor;
pub use readout::Readout;
pub use tensor::Tensor;
pub use types::{IOType, StorageType};
