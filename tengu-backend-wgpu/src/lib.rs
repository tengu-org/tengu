//! This crate provides an implementation of the `tengu-backend` using WGPU, a modern graphics API for GPU-based computation.
//! This implementation leverages WGPU to perform efficient tensor operations, compute passes, and data management on the GPU.
//! The crate is structured into several modules, each responsible for a specific aspect of the backend functionality.
//!
//! # Modules
//!
//! - `backend`: Defines the main `Backend` struct that manages the WGPU device and provides methods to manipulate GPU resources.
//! - `compute`: Implements the `Compute` struct, which is used to manage and execute compute passes on the GPU.
//! - `error`: Contains error handling utilities and custom error types used throughout the crate.
//! - `linker`: Implements the `Linker` struct, which handles copying data between GPU buffers.
//! - `probe`: Defines the `Probe` struct for reading back data from the GPU to the CPU.
//! - `processor`: Implements the `Processor` struct, which sets up and manages shader programs and their associated resources.
//! - `readout`: Implements the `Readout` struct, which is responsible for reading out data from the GPU.
//! - `source`: Contains utilities for managing data sources and their buffers.
//! - `tensor`: Defines the `Tensor` struct, which represents a tensor stored on the GPU and provides methods for tensor operations.

pub mod backend;
pub mod compute;
pub mod error;
pub mod linker;
pub mod probe;
pub mod processor;
pub mod readout;
pub mod source;
pub mod tensor;

pub use backend::Backend;
pub use error::*;
