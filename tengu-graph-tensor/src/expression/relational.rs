use tengu_backend::Backend;
use tengu_tensor::{Relation, StorageType, Unify};

use crate::node::Node;

use super::Expression;

pub struct Relational<S, B> {
    relation: Relation,
    shape: Vec<usize>,
    count: usize,
    lhs: Box<dyn Node<S, B>>,
    rhs: Box<dyn Node<S, B>>,
}

impl<S, B> Relational<S, B>
where
    B: Backend + 'static,
    S: StorageType,
{
    /// Creates a new `Binary` instance.
    ///
    /// # Parameters
    /// - `operator`: The binary operator to apply.
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Binary` instance.
    ///
    /// # Panics
    /// Panics if the shapes of `lhs` and `rhs` do not match.
    fn new<U1: StorageType, U2: StorageType>(
        relation: Relation,
        lhs: Expression<S, U1, B>,
        rhs: Expression<S, U2, B>,
    ) -> Self {
        let shape = lhs.shape().unify(rhs.shape()).expect("Shapes don't match");
        let count = shape.iter().product();
        Self {
            relation,
            shape,
            count,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        }
    }
}

// NOTE: Relational expression constructors.

impl<T, B> Relational<T, B>
where
    B: Backend + 'static,
    T: StorageType,
{
    /// Creates a new `Binary` instance for equality comparison.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Expression` instance with the equality comparison operation.
    pub fn eq<U1: StorageType, U2: StorageType>(
        lhs: Expression<T, U1, B>,
        rhs: Expression<T, U2, B>,
    ) -> Expression<bool, T, B> {
        Expression::Relational(Relational::new(Relation::Eq, lhs, rhs))
    }

    /// Creates a new `Binary` instance for inequality comparison.
    ///
    /// # Parameters
    /// - `lhs`: The left-hand side tensor expression.
    /// - `rhs`: The right-hand side tensor expression.
    ///
    /// # Returns
    /// A new `Expression` instance with the inequality comparison operation.
    pub fn neq<U1: StorageType, U2: StorageType>(
        lhs: Expression<T, U1, B>,
        rhs: Expression<T, U2, B>,
    ) -> Expression<bool, T, B> {
        Expression::Relational(Relational::new(Relation::Neq, lhs, rhs))
    }
}
//
// NOTE: Node implementation.

impl<T, B> Node<T, B> for Relational<T, B>
where
    B: Backend + 'static,
    T: StorageType,
{
    /// Returns the maximum number of elements in lhs and rhs subexpressions.
    ///
    /// # Returns
    /// The number of elements in the tensor.
    fn count(&self) -> usize {
        self.count
    }

    /// Returns the shape of the tensor expression as a slice of dimensions.
    /// This shape is the result of unification on dimensions of lhs and rhs subexpressions.
    ///
    /// # Returns
    /// A slice representing the dimensions of the tensor.
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns a boxed clone of the `Binary` instance.
    ///
    /// # Returns
    /// A boxed clone of the `Binary` instance.
    fn clone_box(&self) -> Box<dyn Node<T, B>> {
        Box::new(self.clone())
    }
}

// NOTE: Clone implementation.

impl<T, B> Clone for Relational<T, B>
where
    B: Backend + 'static,
    T: StorageType,
{
    /// Creates a clone of the `Binary` instance.
    ///
    /// # Returns
    /// A clone of the `Binary` instance.
    fn clone(&self) -> Self {
        Self {
            relation: self.relation,
            shape: self.shape.clone(),
            count: self.count,
            lhs: self.lhs.clone_box(),
            rhs: self.rhs.clone_box(),
        }
    }
}
