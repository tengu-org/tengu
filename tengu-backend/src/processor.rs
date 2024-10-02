use crate::{Backend, StorageType};

pub trait Processor<'a> {
    /// Type of the backend.
    type Backend: Backend;

    /// Type of the representation produced and accepted by this processor.
    type Repr;

    fn var<T: StorageType>(&mut self, tensor: &'a <Self::Backend as Backend>::Tensor<T>) -> Self::Repr;

    fn scalar<T: StorageType>(&mut self, value: T) -> Self::Repr;

    fn unary_fn(&mut self, inner: Self::Repr, symbol: &str) -> Self::Repr;

    fn binary(&mut self, lhs: Self::Repr, rhs: Self::Repr, symbol: &str) -> Self::Repr;

    fn cast(&mut self, inner: Self::Repr, ty: &str) -> Self::Repr;

    fn statement(&mut self, out: Self::Repr, expr: Self::Repr) -> Self::Repr;

    fn block(&mut self, exprs: impl Iterator<Item = Self::Repr>);
}
