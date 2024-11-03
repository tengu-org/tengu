/// Trait for unary functions, to be implemented by backend tensors.
pub trait UnaryFn {
    /// Applies the exponential function to the tensor.
    fn exp(&self) -> Self;

    /// Applies the natural logarithm function to the tensor.
    fn log(&self) -> Self;
}
