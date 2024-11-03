/// Trait for arithmetic operations, to be implemented by backend tensors.
pub trait Arithmetic {
    fn add(&self, other: &Self) -> Self;

    fn sub(&self, other: &Self) -> Self;

    fn mul(&self, other: &Self) -> Self;

    fn div(&self, other: &Self) -> Self;
}
