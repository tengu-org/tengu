//! This module defines the `Probe` trait, which is used for retrieving data in an asynchronous manner
//! from different types that implement the `IOType` trait. The `Probe` trait provides an interface for
//! extracting data from a tensor buffer.

#![allow(async_fn_in_trait)]

use std::borrow::Cow;

use crate::{IOType, Result};

/// The `Probe` trait defines an interface for retrieving data asynchronously
/// from the associated tensor.
///
/// # Type Parameters
/// - `T`: The type of data to be retrieved, which must implement the `IOType` trait.
pub trait Probe<T: IOType> {
    /// Asynchronously retrieves the data from the probe.
    ///
    /// # Returns
    /// A reference or an owned copy of the retrieved data if there are no errors. Otherwise,
    /// an error is returned.
    ///
    /// # Errors
    /// This method may return an error if the data retrieval process encounters any issues.
    async fn retrieve(&self) -> Result<Cow<'_, [T]>>;
}
