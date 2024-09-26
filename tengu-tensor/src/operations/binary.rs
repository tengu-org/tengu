use std::sync::Arc;

use wgpu::Buffer;

use crate::Tengu;

use super::ENTRY;

pub async fn execute_shader(
    tengu: Arc<Tengu>,
    a_buffer: &Buffer,
    b_buffer: &Buffer,
    output_buffer: &Buffer,
    size: usize,
    shader_source: &str,
) {
    // Create a compute pipeline
    let pipeline_layout = tengu.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = tengu
        .device()
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: ENTRY,
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

    // Submit the commands
    tengu.device().compute(|encoder| {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups((size as u32 + 63) / 64, 1, 1); // Assumes workgroup_size is 64
    });
}
