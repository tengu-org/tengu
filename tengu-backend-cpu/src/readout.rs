use tengu_backend::{Backend, Result};
use tracing::trace;

use crate::Backend as CPUBackend;

pub struct Readout;

impl tengu_backend::Readout<'_> for Readout {
    type Backend = CPUBackend;

    async fn commit(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>) -> Result<()> {
        trace!("Comitting readout operation...");
        processor.sources().for_each(|tensor| tensor.readout());
        Ok(())
    }
}
