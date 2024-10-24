/// Operator module defines the `Operator` enum that represents binary operators that can be used in tensor expressions.

/// Operator enum defines possible binary operators that can be used in tensor expressions.
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
