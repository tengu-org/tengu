//! This crate provdes the `Tensor` struct and associated methods for managing tensor objects. It
//! is separated from the backend implementation of tensors and focuses only on user-facing side.
//! All tensor operations
mod channel;
mod error;
mod probe;
mod tensor;

pub use error::{Error, Result};
pub use probe::Probe;
pub use tensor::Tensor;
