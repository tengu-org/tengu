//! This module provides functionality for creating and managing compute pipelines, layouts, and bind groups in the WGPU backend.
//! These constructs are essential for organizing and executing GPU compute workloads efficiently.
//!
//! ## Pipelines in WGPU
//!
//! Pipelines in WGPU are used to configure and manage the GPU's pipeline state. A compute pipeline specifically handles the execution
//! of compute shaders.
//! The `Pipeline` struct in this module encapsulates a compute pipeline and its associated bind group, allowing for efficient execution
//! of compute tasks.
//! - `Pipeline::new`: Creates a new compute pipeline with the given `wgpu::ComputePipeline` and `wgpu::BindGroup`.
//! - `Pipeline::bind_group`: Returns a reference to the bind group associated with the pipeline.
//!
//! ## Layouts in WGPU
//!
//! Layouts in WGPU define how resources such as buffers and textures are bound to the GPU pipeline. A bind group layout specifies the
//! types and visibility of resources that can be accessed by shaders.
//! The `LayoutBuilder` struct in this module provides a builder pattern for creating bind group layouts and their associated bind
//! groups.
//! - `LayoutBuilder::new`: Creates a new layout builder for the specified device.
//! - `LayoutBuilder::add_entry`: Adds a single buffer entry to the layout and bind group.
//! - `LayoutBuilder::add_entries`: Adds multiple buffer entries to the layout and bind group.
//! - `LayoutBuilder::pipeline`: Finalizes the layout and bind group, and returns a `PipelineBuilder` for creating a compute pipeline.
//!
//! ## Bind Groups in WGPU
//!
//! Bind groups in WGPU are collections of resources that are bound together for use by shaders. They are created from bind group
//! layouts and provide the actual bindings for resources.
//! This module includes methods for creating bind group entries and layouts, as well as constructing bind groups themselves.
//! - `create_layout_entry`: Helper function to create a bind group layout entry for a buffer.
//! - `create_bind_entry`: Helper function to create a bind group entry for a buffer.
//!
//! ## Module Structs and Methods
//!
//! - `Pipeline`: Represents a compute pipeline in WGPU, encapsulating a compute pipeline and its associated bind group.
//! - `LayoutBuilder`: Provides a builder pattern for creating bind group layouts and their associated bind groups.
//! - `PipelineBuilder`: Provides a builder pattern for creating compute pipelines, encapsulating the pipeline layout and bind group.
//!   - `PipelineBuilder::new`: Creates a new pipeline builder with the specified device, bind group, and bind group layout.
//!   - `PipelineBuilder::with_label`: Sets a label for the pipeline.
//!   - `PipelineBuilder::build`: Builds the compute pipeline with the specified shader module.

use std::ops::Deref;

use crate::{Buffer, BufferUsage, Device};

const ENTRY: &str = "main";

// NOTE: Compute pipeline

/// Represents a compute pipeline in the WGPU backend.
pub struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl Pipeline {
    /// Creates a new `Pipeline` instance.
    ///
    /// # Parameters
    /// - `pipeline`: The WGPU compute pipeline.
    /// - `bind_group`: The bind group associated with the pipeline.
    ///
    /// # Returns
    /// A new `Pipeline` instance.
    pub fn new(pipeline: wgpu::ComputePipeline, bind_group: wgpu::BindGroup) -> Self {
        Self { pipeline, bind_group }
    }
    /// Returns a reference to the bind group associated with the pipeline.
    ///
    /// # Returns
    /// A reference to the `wgpu::BindGroup`.
    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

impl Deref for Pipeline {
    type Target = wgpu::ComputePipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

// NOTE: LayoutBuilder implementation

/// Builder for creating a pipeline layout and bind group.
pub struct LayoutBuilder<'a, 'device> {
    device: &'device Device,
    buffers: Vec<&'a Buffer>,
    layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    bind_entries: Vec<wgpu::BindGroupEntry<'a>>,
    counter: usize,
}

impl<'a, 'device> LayoutBuilder<'a, 'device> {
    /// Creates a new `LayoutBuilder` instance.
    ///
    /// # Parameters
    /// - `device`: The device to use for creating the layout and bind group.
    ///
    /// # Returns
    /// A new `LayoutBuilder` instance.
    pub fn new(device: &'device Device) -> Self {
        Self {
            device,
            buffers: Vec::new(),
            layout_entries: Vec::new(),
            bind_entries: Vec::new(),
            counter: 0,
        }
    }

    /// Adds a buffer entry to the layout and bind group.
    ///
    /// # Parameters
    /// - `buffer`: The buffer to add.
    ///
    /// # Returns
    /// The updated `LayoutBuilder`.
    pub fn add_entry(mut self, buffer: &'a Buffer) -> Self {
        self.layout_entries.push(create_layout_entry(buffer, self.counter));
        self.bind_entries.push(create_bind_entry(buffer, self.counter));
        self.buffers.push(buffer);
        self.counter += 1;
        self
    }

    /// Adds multiple buffer entries to the layout and bind group.
    ///
    /// # Parameters
    /// - `buffers`: An iterator over the buffers to add.
    ///
    /// # Returns
    /// The updated `LayoutBuilder`.
    pub fn add_entries(mut self, buffers: impl IntoIterator<Item = &'a Buffer>) -> Self {
        for buffer in buffers.into_iter() {
            self.layout_entries.push(create_layout_entry(buffer, self.counter));
            self.bind_entries.push(create_bind_entry(buffer, self.counter));
            self.buffers.push(buffer);
            self.counter += 1;
        }
        self
    }
    /// Creates a `PipelineBuilder` for further configuring and building the compute pipeline.
    ///
    /// # Parameters
    /// - `label`: A label for the bind group and layout.
    ///
    /// # Returns
    /// A `PipelineBuilder` instance.
    pub fn pipeline(self, label: &str) -> PipelineBuilder<'device> {
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &self.layout_entries,
        });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &bind_group_layout,
            entries: &self.bind_entries,
        });
        PipelineBuilder::new(self.device, bind_group, bind_group_layout)
    }
}

/// Creates a bind group layout entry for a buffer.
///
/// # Parameters
/// - `buffer`: The buffer to create the layout entry for.
/// - `idx`: The binding index.
///
/// # Returns
/// A `wgpu::BindGroupLayoutEntry`.
fn create_layout_entry(buffer: &Buffer, idx: usize) -> wgpu::BindGroupLayoutEntry {
    let read_only = match buffer.usage() {
        BufferUsage::Read => true,
        BufferUsage::Write => false,
        BufferUsage::ReadWrite => false,
        BufferUsage::Staging => panic!("staging buffers should not belong to a bind group"),
    };
    wgpu::BindGroupLayoutEntry {
        binding: idx as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
/// Creates a bind group entry for a buffer.
///
/// # Parameters
/// - `buffer`: The buffer to create the bind entry for.
/// - `idx`: The binding index.
///
/// # Returns
/// A `wgpu::BindGroupEntry`.
fn create_bind_entry(buffer: &Buffer, idx: usize) -> wgpu::BindGroupEntry {
    wgpu::BindGroupEntry {
        binding: idx as u32,
        resource: buffer.as_entire_binding(),
    }
}

// NOTE: PipelineBuilder implementation

/// Builder for creating and configuring a compute pipeline.
pub struct PipelineBuilder<'device> {
    device: &'device Device,
    label: Option<String>,
    layout: wgpu::PipelineLayout,
    bind_group: wgpu::BindGroup,
}

impl<'device> PipelineBuilder<'device> {
    /// Creates a new `PipelineBuilder` instance.
    ///
    /// # Parameters
    /// - `device`: The device to use for creating the pipeline.
    /// - `bind_group`: The bind group to use in the pipeline.
    /// - `bind_group_layout`: The layout of the bind group.
    ///
    /// # Returns
    /// A new `PipelineBuilder` instance.
    pub fn new(device: &'device Device, bind_group: wgpu::BindGroup, bind_group_layout: wgpu::BindGroupLayout) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        Self {
            device,
            label: None,
            layout: pipeline_layout,
            bind_group,
        }
    }

    /// Sets a label for the compute pipeline.
    ///
    /// # Parameters
    /// - `label`: The label to set.
    ///
    /// # Returns
    /// The updated `PipelineBuilder`.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Builds the compute pipeline with the specified shader module.
    ///
    /// # Parameters
    /// - `shader`: The shader module to use in the pipeline.
    ///
    /// # Returns
    /// A `Pipeline` instance.
    pub fn build(self, shader: wgpu::ShaderModule) -> Pipeline {
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: self.label.as_deref(),
            layout: Some(&self.layout),
            module: &shader,
            entry_point: ENTRY,
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });
        Pipeline::new(pipeline, self.bind_group)
    }
}
