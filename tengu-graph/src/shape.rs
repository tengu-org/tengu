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
