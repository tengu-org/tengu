//! This crate provdes the `Tensor` struct and associated methods for managing tensor objects. It
//! is separated from the backend implementation of tensors and focuses only on user-facing side.
//! All tensor operations that are implemented in this crate rely on the particular backend
//! implementation, this module is essnetially a wrapper around backend tensors. However, one piece
//! of functionality it provides is the ability to create probes that can asynchronously retrieve
//! data from tensors.
//!
//! ## Features
//!
//! - Tensor Management: Defines the Tensor struct for managing tensor objects with support for different backends.
//! - Data Inspection: The Probe struct allows users to asynchronously inspect and retrieve tensor data for analysis and debugging.
//! - Channel Communication: Implements a sender-receiver pattern for transferring data between tensors and probes.
//!
//! ## Modules
//! - `channel`: Defines the `Channel` struct for managing asynchronous data retrieval from tensors.
//! - `error`: Defines the `Error` and `Result` types for error handling.
//! - `probe`: Defines the `Probe` struct for asynchronously retrieving data from tensors.
//! - `tensor`: Defines the `Tensor` struct for managing tensor objects.

mod channel;
mod error;
mod probe;
mod tensor;

pub use error::{Error, Result};
pub use probe::Probe;
pub use tensor::Tensor;
