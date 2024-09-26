use crate::Device;

const ENTRY: &str = "main";

// Builder implementation

pub struct ComputePipeline<'device> {
    device: &'device Device,
    layout_label: Option<String>,
    pipeline_label: Option<String>,
    layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl<'device> ComputePipeline<'device> {
    pub fn new(device: &'device Device, layout: wgpu::BindGroupLayout, bind_group: wgpu::BindGroup) -> Self {
        Self {
            device,
            layout_label: None,
            pipeline_label: None,
            layout,
            bind_group,
        }
    }

    pub fn with_layout_label(mut self, label: impl Into<String>) -> Self {
        self.layout_label = Some(label.into());
        self
    }

    pub fn with_pipeline_label(mut self, label: impl Into<String>) -> Self {
        self.pipeline_label = Some(label.into());
        self
    }

    pub fn build(&self, shader: wgpu::ShaderModule) -> wgpu::ComputePipeline {
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: self.layout_label.as_deref(),
            bind_group_layouts: &[&self.layout],
            push_constant_ranges: &[],
        });
        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: self.pipeline_label.as_deref(),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: ENTRY,
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        })
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}
