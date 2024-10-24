use as_any::{AsAny, Downcast};
use async_trait::async_trait;

use tengu_backend::{Backend, Linker};
use tengu_backend_tensor::StorageType;
use tengu_graph_tensor::Tensor;

use crate::shape::Shape;
use crate::{Error, Result};

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
    async fn retrieve(&self) -> Result<()>;
}

// NOTE: Tensor implementation.

#[async_trait(?Send)]
impl<T: StorageType, B: Backend + 'static> Source<B> for Tensor<T, B> {
    /// Retrieves the label of the source.
    ///
    /// # Returns
    /// The label of the source.
    fn label(&self) -> &str {
        self.label()
    }

    /// Checks if the tensor matches the shape of another tensor.
    ///
    /// # Parameters
    /// - `other`: Another source to compare against.
    ///
    /// # Returns
    /// A result indicating whether the shapes match.
    fn matches_to(&self, other: &dyn Source<B>) -> Result<bool> {
        let other = other.downcast_ref::<Self>().ok_or_else(|| Error::TypeMismatch)?;
        Ok(self.shape() == other.shape())
    }

    /// Copies the data from this tensor to another tensor using the provided linker.
    ///
    /// # Parameters
    /// - `to`: The target tensor to link to.
    /// - `linker`: The linker to use for copying the link.
    ///
    /// # Returns
    /// A result indicating the success of the operation.
    fn copy(&self, to: &dyn Source<B>, linker: &mut B::Linker<'_>) -> Result<()> {
        let to = to.downcast_ref::<Self>().ok_or_else(|| Error::TypeMismatch)?;
        linker.copy_link(self.raw(), to.raw());
        Ok(())
    }

    /// Reads the tensor data from the source and sends it to associated probes.
    /// If the channel is full then there is no point wasting time on reading the data out - the
    /// previous message hasn't been read out by the probe yet. In this case the method will return
    /// immediately without retrieving and sending anything.
    ///
    /// # Returns
    /// A result indicating the success of the operation.
    async fn retrieve(&self) -> Result<()> {
        self.retrieve().await.map_err(Error::TensorError)
    }
}
