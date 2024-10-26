//! This crate provides an implementation of the `tengu-backend` using CPU. The CPU implementation
//! is notoriously slow. While there are some optimizations here and there, it is slow and uses
//! many unnecessary allocations. While optimizations are possible and might be introduced in the
//! future, the main goal of the CPU backend is to serve as a reference implementation of tensor
//! operations, for benchmarking, and for testing other (GPU-based) implementations against this
//! one.
//!
//! # Imporatnt modules
//!
//! - `backend`: Defines the main `Backend` struct that manages the CPU-specific subsystems for tensor computations.
//! - `limits`: Defines the `Limits` struct, which holds information about the limits for tensor computations.
//! - `linker`: Implements the `Linker` struct, which handles copying data between tensors.
//! - `processor`: Implements the `Processor` struct, which performs all tensor computations on the CPU.
//! - `source`: Defines the `Source` struct, which stores tensor data in type-independent fasioh.
//! - `tensor`: Defines the `Tensor` struct, which represents a tensor and provides methods for tensor operations.

mod backend;
mod compute;
mod limits;
mod linker;
mod processor;
mod readout;

pub use backend::Backend;
