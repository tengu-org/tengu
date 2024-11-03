use crate::Tensor;

pub trait Relational {
    type Output: Tensor<bool>;

    fn eq(&self, other: &Self) -> Self::Output;

    fn neq(&self, other: &Self) -> Self::Output;
}
