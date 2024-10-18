use std::sync::Arc;
use tengu_backend::{Error, IOType, Result, StorageType, Tensor};
use tracing::trace;

use crate::limits::Limits as CPULimits;
use crate::linker::Linker as CPULinker;
use crate::probe::Probe as CPUProbe;
use crate::processor::Processor as CPUProcessor;
use crate::readout::Readout as CPUReadout;
use crate::tensor::Tensor as CPUTensor;

/// The `Backend` struct is responsible for managing the WGPU device and providing methods to create and manipulate GPU resources.
pub struct Backend;

// NOTE: tengu_backend::Backend implementation

impl tengu_backend::Backend for Backend {
    type Tensor<T: StorageType> = CPUTensor<T>;
    type Compute<'a> = WGPUCompute<'a>;
    type Processor<'a> = CPUProcessor<'a>;
    type Linker<'a> = CPULinker;
    type Readout<'a> = CPUReadout;
    type Limits = CPULimits;

    async fn new() -> tengu_backend::Result<Arc<Self>> {
        trace!("Created CPU instance for backend");
        Ok(Arc::new(Self))
    }

    fn limits(&self) -> Self::Limits {
        CPULimits
    }

    fn processor(&self) -> Self::Processor<'_> {
        CPUProcessor::new()
    }

    fn propagate(&self, call: impl FnOnce(Self::Linker<'_>)) {
        trace!("Executing propagation step...");
        call(CPULinker);
    }

    fn compute<F>(&self, label: &str, call: F) -> Result<()>
    where
        F: FnOnce(Self::Compute<'_>) -> Result<()>,
    {
        trace!("Executing compute step...");
        let commands = self
            .device
            .encoder(label)
            .pass(label, |pass| {
                call(WGPUCompute::new(&self.device, label, pass)).map_err(Into::into)
            })
            .map_err(|e| Error::WGPUError(e.into()))?
            .finish();
        self.device.submit(commands);
        Ok(())
    }

    fn readout(&self, _label: &str, call: impl FnOnce(Self::Readout<'_>)) {
        trace!("Executing readout step...");
        call(CPUReadout);
    }

    fn tensor<T: IOType>(self: &Arc<Self>, label: impl Into<String>, data: &[T]) -> Self::Tensor<T> {
        let label = label.into();
        trace!("Creating new tensor '{label}'...");
        CPUTensor::with_data(self, label, data)
    }

    fn zero<T: StorageType>(self: &Arc<Self>, label: impl Into<String>, count: usize) -> Self::Tensor<T> {
        let label = label.into();
        trace!("Creating new zero tensor '{label}'...");
        CPUTensor::new(self, label, count)
    }
}
