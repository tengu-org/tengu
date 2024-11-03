//! This module defines the `Compute` struct which implements the `Compute` trait from the `tengu_backend` crate.
//! The `Compute` struct is responsible for managing and executing compute passes on the GPU using the WGPU backend.
//!
//! # Role of `Compute` Struct
//! The `Compute` struct plays a crucial role in setting up and executing compute operations on the GPU. It manages the
//! device and compute pass, and it is responsible for creating the pipeline and setting the necessary resources for the
//! compute operations.
//!
//! # Usage
//! The `Compute` struct is used to create the pipeline and bind the necessary resources to the compute pass. This is done
//! in the `commit` method, which performs the following operations:
//! 1. **Pipeline Creation**: The `pipeline` method is called to create a pipeline for the compute operations. This method
//!    generates a shader from the processor's shader code and sets up the necessary buffer bindings.
//! 2. **Resource Binding**: The `run` method sets the pipeline and bind group for the compute pass using the created
//!    pipeline. It binds the resources required for the compute operations.
//! 3. **Dispatch Workgroups**: The `run` method then dispatches the workgroups to execute the compute operations on the GPU.

use std::rc::Rc;

use tengu_backend::Compute as RawCompute;
use tengu_backend::Pass as RawPass;
use tengu_backend::{Error, Operation, Result};
use tengu_utils::Label;
use tengu_wgpu::{Device, Pipeline};
use tracing::trace;

use crate::processor::compute::Processor;
use crate::Backend as WGPUBackend;

const WORKGROUP_SIZE: u32 = 64;

// NOTE: Compute implementation.

/// The `Compute` struct is used to manage and execute compute passes on the GPU. A new `Compute`
/// struct is create for each execution of the commit pass.
pub struct Compute {
    device: Rc<Device>,
    label: Label,
}

impl Operation<WGPUBackend> for Compute {
    type IR<'a> = IR;
    type Pass<'a> = Pass<'a>;

    fn new(backend: &Rc<WGPUBackend>, label: impl Into<Label>) -> Self {
        Self {
            device: Rc::clone(backend.device()),
            label: label.into(),
        }
    }

    fn run<F>(&mut self, call: F) -> Result<()>
    where
        F: FnOnce(Self::Pass<'_>) -> anyhow::Result<()>,
    {
        trace!("Executing compute step");
        let commands = self
            .device
            .encoder(self.label.as_ref())
            .pass(self.label.as_ref(), |pass| call(Pass::new(pass)))
            .map_err(|e| Error::ComputeError(e.into()))?
            .finish();
        trace!("Submitting compute commands to the queue");
        self.device.submit(commands);
        Ok(())
    }
}

impl RawCompute<WGPUBackend> for Compute {
    type Processor<'a> = Processor<'a> where Self: 'a;

    fn processor(&self) -> Self::Processor<'_> {
        Processor::new(&self.device)
    }
}

// NOTE: Pass implementation.

pub struct Pass<'a> {
    pass: wgpu::ComputePass<'a>,
}

impl<'a> Pass<'a> {
    pub fn new(pass: wgpu::ComputePass<'a>) -> Self {
        Self { pass }
    }
}

impl<'a> RawPass<WGPUBackend> for Pass<'a> {
    type IR<'b> = IR;

    /// Runs the compute operations by setting up the pipeline, bind group, and dispatching workgroups.
    ///
    /// # Parameters
    /// - `processor`: A reference to the processor which provides the necessary data for the compute operations.
    ///
    /// # Returns
    /// A `Result` indicating whether the compute operations were successful or an error occurred.
    fn run(&mut self, ir: &Self::IR<'_>) -> Result<()> {
        trace!("Executing compute operation");
        let workgroup_count = ir.count as u32 / WORKGROUP_SIZE + 1;
        self.pass.set_pipeline(&ir.pipeline);
        self.pass.set_bind_group(0, ir.pipeline.bind_group(), &[]);
        self.pass.dispatch_workgroups(workgroup_count, 1, 1);
        trace!("Dispatched workgroups");
        Ok(())
    }
}

// NOTE: Compute state implementation.

pub struct IR {
    pipeline: Pipeline,
    count: usize,
}

impl IR {
    pub fn new(pipeline: Pipeline, count: usize) -> Self {
        Self { pipeline, count }
    }
}
