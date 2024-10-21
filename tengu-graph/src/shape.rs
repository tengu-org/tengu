use tengu_backend::Backend;
use tengu_backend_tensor::StorageType;
use tengu_tensor::Tensor;

/// A trait for types that have a shape.
///
/// The `Shape` trait defines methods for retrieving the shape and the number of elements of a
/// tensor or tensor expression.
pub trait Shape {
    /// Returns the shape of the object.
    ///
    /// # Returns
    /// A slice representing the shape of the object.
    fn shape(&self) -> &[usize];

    /// Returns the number of elements in the object.
    ///
    /// # Returns
    /// The number of elements.
    fn count(&self) -> usize;
}

// NOTE: Tensor implementation.

impl<T: StorageType, B: Backend> Shape for Tensor<T, B> {
    /// Returns the shape of the tensor.
    ///
    /// # Returns
    /// A slice representing the shape of the tensor.
    fn shape(&self) -> &[usize] {
        use tengu_backend_tensor::Tensor;
        self.raw().shape()
    }

    /// Returns the number of elements in the tensor.
    ///
    /// # Returns
    /// The number of elements in the tensor.
    fn count(&self) -> usize {
        use tengu_backend_tensor::Tensor;
        self.raw().count()
    }
}
