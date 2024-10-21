//! This module provides various operations for tensor expressions, including arithmetic and relational operations.
//!
//! The submodules include:
//!
//! - `arithmetic`: Defines arithmetic operations such as addition, subtraction, multiplication, and division for tensor expressions.
//! - `relational`: Defines relational operations such as equality for tensor expressions.

mod arithmetic;
mod relational;

pub use super::binary::Binary;
pub use super::Expression;
