use tengu_backend::Result;

use crate::Backend;

pub struct Compute;

impl tengu_backend::Compute for Compute {
    type Backend = Backend;

    fn run(&mut self, _processor: &<Self::Backend as tengu_backend::Backend>::Processor<'_>) -> Result<()> {
        Ok(())
    }
}
