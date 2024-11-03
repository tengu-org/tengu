use std::rc::Rc;

use crate::{Backend, Result};

mod compute;
mod propagate;
mod readout;

pub use compute::Compute;
pub use propagate::Propagate;
pub use readout::Readout;
use tengu_utils::Label;

pub trait Operation<B: Backend> {
    type IR<'a>;
    type Pass<'a>: Pass<B, IR<'a> = Self::IR<'a>>;

    fn new(backend: &Rc<B>, label: impl Into<Label>) -> Self;

    fn run<F>(&mut self, call: F) -> Result<()>
    where
        F: FnOnce(Self::Pass<'_>) -> anyhow::Result<()>;
}

pub trait Pass<B: Backend> {
    type IR<'a>;

    /// Uses the given processor state to perform the operation.
    ///
    /// # Parameters
    /// - `ir`: A reference to the processor that was used to prepare data from operation.
    ///
    /// # Returns
    /// A `Result` indicating whether the operation was successful or an error occurred.
    fn run(&mut self, ir: &Self::IR<'_>) -> Result<()>;
}
