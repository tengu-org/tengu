/// Enum representing supported unary functions.
#[derive(Copy, Clone)]
pub enum Function {
    /// Logarithm function.
    Log,
    /// Exponentiation function.
    Exp,
}

pub trait UnaryFn {
    fn exp(&self) -> Self;
    fn log(&self) -> Self;
}
