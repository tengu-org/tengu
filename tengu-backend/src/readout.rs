use crate::Backend;

pub trait Readout<'a> {
    /// Type of the backend.
    type Backend: Backend;

    fn commit(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>);
}
