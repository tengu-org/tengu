//! This module defines the `Probe` trait, which is used for retrieving data in an asynchronous manner
//! from different types that implement the `IOType` trait. The `Probe` trait provides an interface for
//! extracting data from a tensor buffer.
//!
#![allow(async_fn_in_trait)]

use crate::{IOType, Result};

/// The `Probe` trait defines an interface for retrieving data asynchronously
/// from the associated tensor. The retrieved data is stored in the provided buffer.
///
/// # Type Parameters
/// - `T`: The type of data to be retrieved, which must implement the `IOType` trait.
pub trait Probe<T: IOType> {
    /// Asynchronously retrieves the data from the probe and stores it in the provided buffer.
    ///
    /// # Parameters
    /// - `buffer`: A mutable reference to a `Vec` where the retrieved data will be stored.
    ///
    /// # Returns
    /// A `Result` type that indicates the success or failure of the operation.
    ///
    /// # Errors
    /// This method may return an error if the data retrieval process encounters any issues.
    async fn retrieve_to(&self, buffer: &mut Vec<T>) -> Result<()>;
}
