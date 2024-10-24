/// This module defines the `Function` enumeration, which represents the supported unary functions
/// that can be applied to tensors. It also defines the `UnaryFn` trait, which is to be implemented
/// by backend tensors.

/// Enum representing supported unary functions.
#[derive(Copy, Clone)]
pub enum Function {
    /// Logarithm function.
    Log,
    /// Exponentiation function.
    Exp,
}

/// Trait for unary functions, to be implemented by backend tensors.
pub trait UnaryFn {
    /// Applies the exponential function to the tensor.
    fn exp(&self) -> Self;

    /// Applies the natural logarithm function to the tensor.
    fn log(&self) -> Self;
}
