//! This crate defines an interface for implementing tensor operations on various backends. It is designed
//! to be flexible and extensible, allowing for different types of backends such as CPU, Vulkan, or any other
//! backend. This crate provides the foundational traits and types that other backend implementations must
//! adhere to in order to support tensor operations. The crate is intended to be consumed by user-facing
//! crates such as `tengu-tensor`.
//!
//! ## Features
//!
//! - Backend Abstraction: Provides the Backend trait, allowing support for different hardware and software environments.
//! - Compute Operations: Provides the Compute trait for executing tasks within a backend using preprocessed tensor data.
//! - Data Linking: The Linker trait facilitates the propagation of tensor data within computation graphs.
//! - Processor Management: The Processor trait manages the AST of tensor expressions and prepares data for backend use.
//!
//! ## Modules
//!
//! - `backend`: Defines the `Backend` trait, which serves as the main interface for tensor computation backends.
//!   This trait includes methods for creating and managing tensors, processors, compute instances, linkers,
//!   readouts, and retieves.
//! - `compute`: Defines the `Compute` trait, which provides an interface for performing computations
//!   using a given processor state within a specified backend.
//! - `error`: Defines error handling types used throughout the crate. It includes the `Error` type
//!   and the `Result` type alias for simplifying error handling.
//! - `linker`: Defines the `Linker` trait, which is used for propagating tensor data between different parts
//!   of a computation graph or between different storage locations.
//! - `processor`: Defines the `Processor` trait, which is used for processing the abstract syntax tree (AST)
//!   of tensor expressions in a final tagless style and preparaing the results for backend
//!   consumption.
//! - `readout`: Defines the `Readout` trait, which provides an interface for reading out tensor data
//!   from a computation graph into the staging buffers associated with tensors.
//! - `limits`: Defines the `Limits` struct, which represents the limits of the backend in terms of
//!   max tensor count and other properties.
//!
//! ## Traits
//!
//! Implementors of this crate will need to implement the following traits to create a functional backend:
//!
//! - **Backend**: The core trait that defines the overall interface for tensor computation backends.
//!   It includes associated types and methods for creating and managing tensors, processors, compute instances,
//!   linkers, readouts, and retrieves.
//! - **Compute**: Defines how computations are committed and executed using a given processor state within the backend.
//! - **Linker**: Manages the propagation of tensor data between different  parts of the computation graph.
//! - **Processor**: Handles the processing of the abstract syntax tree (AST) for tensor expressions.
//! - **Readout**: Provides methods for reading tensor data from the computation graph into staging
//!   buffers.

mod backend;
mod compute;
mod error;
mod limits;
mod linker;
mod processor;
mod readout;

pub use backend::Backend;
pub use compute::Compute;
pub use error::{Error, Result};
pub use limits::Limits;
pub use linker::Linker;
pub use processor::Processor;
pub use readout::Readout;
