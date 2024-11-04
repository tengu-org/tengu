use tengu_backend::Backend;
use tengu_tensor::StorageType;

/// A trait for AST nodes in the Tengu framework. Any expression or a tensor is a node.
///
/// The `Node` trait extends the `Shape` trait and defines methods for visiting, finding, and cloning nodes.
pub trait Node<T: StorageType, B: Backend> {
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

    /// Clones the node into a boxed trait object.
    ///
    /// # Returns
    /// A boxed trait object containing the cloned node.
    fn clone_box(&self) -> Box<dyn Node<T, B>>;
}
