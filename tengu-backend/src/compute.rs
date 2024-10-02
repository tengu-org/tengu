use crate::Backend;

pub trait Compute {
    /// The type of backend
    type Backend: Backend;

    /// Uses given processor state to perform computations.
    fn commit(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>);
}
