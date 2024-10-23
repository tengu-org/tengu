#[derive(Clone, Copy)]
pub enum Operator {
    /// Addition operator.
    Add,
    /// Subtraction operator.
    Sub,
    /// Multiplication operator.
    Mul,
    /// Division operator.
    Div,
    /// Equality operator.
    Eq,
    /// Inequality operator.
    Neq,
}
