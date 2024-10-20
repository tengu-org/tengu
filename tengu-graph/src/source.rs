use as_any::AsAny;
use async_trait::async_trait;
use tengu_backend::Backend;

use crate::Result;

/// A trait for tensors to treat the uniformly irrespective of their underlying type.
///
/// The `Source` trait defines methods for matching to and copying links between sources.
#[async_trait(?Send)]
pub trait Source<B: Backend>: AsAny {
    /// Retrieves the label of the source.
    ///
    /// # Returns
    /// The label of the source.
    fn label(&self) -> &str;

    /// Checks if the source matches another source.
    ///
    /// # Parameters
    /// - `other`: The other source to match against.
    ///
    /// # Returns
    /// A result containing a boolean indicating whether the sources match.
    fn matches_to(&self, other: &dyn Source<B>) -> Result<bool>;

    /// Copies a source tensor to another source.
    ///
    /// # Parameters
    /// - `to`: The destination source to copy the tensor to.
    /// - `linker`: A mutable reference to the linker.
    ///
    /// # Returns
    /// A result indicating success or failure.
    fn copy(&self, to: &dyn Source<B>, linker: &mut B::Linker<'_>) -> Result<()>;

    /// Retrieves data from the source tensor and sends it to all associated probes.
    ///
    /// # Returns
    /// A result indicating success or failure.
    async fn readout(&self) -> Result<()>;
}
