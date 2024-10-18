//! This module defines the `Source` trait which is used to represent a data source that can provide
//! a buffer, a label, and a method to read data using an encoder. It is needed to treat tensor in
//! uniform fashion, irrespecitve of their underlying storage type.

use async_trait::async_trait;
use tengu_backend::Result;
use tengu_wgpu::{Buffer, Encoder};

/// The `Source` trait represents a "type-less" tensor. It is implemented only but tensor and used
/// by the `Processor` to handle all tensors in a uniform fashion.
#[async_trait]
pub trait Source {
    /// Returns a label that describes the source.
    ///
    /// # Returns
    /// A string slice that holds the label.
    fn label(&self) -> &str;

    /// Returns a reference to the buffer associated with the source.
    ///
    /// # Returns
    /// A reference to the tengu-wgpu `Buffer` object.
    fn buffer(&self) -> &Buffer;

    /// Performs a readout operation using the provided encoder.
    ///
    /// # Parameters
    /// - `encoder`: A mutable reference to an `Encoder` object used for the readout operation.
    fn readout(&self, encoder: &mut Encoder);

    /// Performs a retrieval operation using the provided encoder.
    ///
    /// # Returns
    /// A `Result` indicating whether the retrieval was successful or an error occurred.
    async fn retrieve(&self) -> Result<()>;

    /// Returns the number of elements in the source.
    ///
    /// # Returns
    /// The number of elements in the source tensor.
    fn count(&self) -> usize;
}
