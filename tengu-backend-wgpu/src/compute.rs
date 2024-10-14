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
//! 2. **Resource Binding**: The `commit` method sets the pipeline and bind group for the compute pass using the created
//!    pipeline. It binds the resources required for the compute operations.
//! 3. **Dispatch Workgroups**: The `commit` method dispatches the workgroups to execute the compute operations on the GPU.

use tengu_backend::{Backend, Error, Result};
use tengu_wgpu::{Device, Pipeline};

use crate::processor::Processor;
use crate::Backend as WGPUBackend;

const WORKGROUP_SIZE: u32 = 64;

/// The `Compute` struct is used to manage and execute compute passes on the GPU. A new `Compute`
/// struct is create for each execution of the commit pass.
pub struct Compute<'a> {
    device: &'a Device,
    label: &'a str,
    pass: wgpu::ComputePass<'a>,
}

impl<'a> Compute<'a> {
    /// Creates a new `Compute` instance.
    ///
    /// # Parameters
    /// - `device`: A reference to the `Device` object used for GPU operations.
    /// - `label`: A label for the compute operations.
    /// - `pass`: A `wgpu::ComputePass` object representing the compute pass.
    ///
    /// # Returns
    /// A new instance of `Compute`.
    pub fn new(device: &'a Device, label: &'a str, pass: wgpu::ComputePass<'a>) -> Self {
        Self { device, label, pass }
    }

    /// Creates a pipeline for the compute operations using the given processor.
    ///
    /// # Parameters
    /// - `processor`: A reference to the `Processor` object which provides shader and buffer information.
    ///
    /// # Returns
    /// A `Result` containing the `Pipeline` object if the pipeline creation is successful, or an `Error` if
    /// the buffer limit is reached.
    fn pipeline(&self, processor: &Processor<'_>) -> Result<Pipeline> {
        let shader = self.device.shader(self.label, processor.shader());
        let buffers = processor.sources().map(|source| source.buffer()).collect::<Vec<_>>();
        let max_buffers = self.device.limits().max_storage_buffers_per_shader_stage as usize;
        if buffers.len() > max_buffers {
            return Err(Error::BufferLimitReached(max_buffers));
        }
        let pipeline = self
            .device
            .layout()
            .add_entries(buffers)
            .pipeline(self.label)
            .build(shader);
        Ok(pipeline)
    }
}

impl<'a> tengu_backend::Compute for Compute<'a> {
    type Backend = WGPUBackend;

    /// Commits the compute operations by setting up the pipeline, bind group, and dispatching workgroups.
    ///
    /// # Parameters
    /// - `processor`: A reference to the processor which provides the necessary data for the compute operations.
    ///
    /// # Returns
    /// A `Result` indicating whether the compute operations were successful or an error occurred.
    fn commit(&mut self, processor: &<Self::Backend as Backend>::Processor<'_>) -> tengu_backend::Result<()> {
        let pipeline = self.pipeline(processor)?;
        let workgroup_count = processor.element_count() as u32 / WORKGROUP_SIZE + 1;
        self.pass.set_pipeline(&pipeline);
        self.pass.set_bind_group(0, pipeline.bind_group(), &[]);
        self.pass.dispatch_workgroups(workgroup_count, 1, 1);
        Ok(())
    }
}
