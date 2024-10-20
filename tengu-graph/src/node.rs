use tengu_backend::{Backend, Processor};

use crate::collector::Collector;
use crate::source::Source;

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

/// A trait for AST nodes in the Tengu framework. Any expression or a tensor is a node.
///
/// The `Node` trait extends the `Shape` trait and defines methods for visiting, finding, and cloning nodes.
pub trait Node<B: Backend>: Shape {
    /// Visits the node with a processor.
    ///
    /// # Parameters
    /// - `processor`: A mutable reference to the processor.
    ///
    /// # Returns
    /// The inner representation of the processor, as defined by `tengu-backend`.
    fn visit<'a>(&'a self, processor: &mut B::Processor<'a>) -> <B::Processor<'a> as Processor>::Repr;

    /// Collect sources from the expression tree.
    ///
    /// # Parameters
    /// - `collector`: A mutable reference to the collector.
    fn collect<'a>(&'a self, collector: &mut Collector<'a, B>);

    /// Finds a source by its label.
    ///
    /// # Parameters
    /// - `label`: The label of the source to find.
    ///
    /// # Returns
    /// An optional reference to the source.
    fn find<'a>(&'a self, label: &str) -> Option<&'a dyn Source<B>>;

    /// Clones the node into a boxed trait object.
    ///
    /// # Returns
    /// A boxed trait object containing the cloned node.
    fn clone_box(&self) -> Box<dyn Node<B>>;
}
