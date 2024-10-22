use crate::Backend;

pub struct Readout;

impl tengu_backend::Readout for Readout {
    type Backend = Backend;

    fn run(&mut self, _processor: &<Self::Backend as tengu_backend::Backend>::Processor<'_>) {
        todo!()
    }
}
