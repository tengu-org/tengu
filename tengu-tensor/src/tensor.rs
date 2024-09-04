use pollster::FutureExt;
use std::ops::Add;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{Buffer, BufferUsages};

use crate::context::Context;

pub struct TensorBuilder<'a> {
    shape: Vec<usize>,
    size: usize,
    context: &'a Context,
}

impl<'a> TensorBuilder<'a> {
    pub fn new(context: &'a Context, shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let size = shape.iter().product();
        Self { shape, size, context }
    }

    pub fn init(self, data: &[f32]) -> Tensor<'a> {
        assert_eq!(data.len(), self.size, "data length does not match shape");
        let buffer = self.context.device().create_buffer_init(&BufferInitDescriptor {
            label: Some("Tensor Buffer"),
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        });
        Tensor {
            buffer,
            size: self.size,
            shape: self.shape,
            context: self.context,
        }
    }
}

pub struct Tensor<'a> {
    buffer: Buffer,
    size: usize,
    shape: Vec<usize>,
    context: &'a Context,
}

impl<'a> Tensor<'a> {
    pub async fn data(&self) -> Vec<f32> {
        let staging_buffer = self.make_staging_buffer();
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = flume::bounded(1);
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.context.device().poll(wgpu::Maintain::wait()).panic_on_timeout();
        if let Ok(Ok(())) = receiver.recv_async().await {
            let data = buffer_slice.get_mapped_range();
            let result = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            result
        } else {
            panic!("failed to retrieve tensor buffer") // TODO: return Result
        }
    }

    pub async fn add(self, other: Tensor<'a>) -> Tensor<'a> {
        assert_eq!(self.shape, other.shape, "Shapes must match for addition");
        let size = (self.size * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let result_buffer = self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Result Buffer"),
            size,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        crate::operations::binary::execute_shader(
            self.context,
            &self.buffer,
            &other.buffer,
            &result_buffer,
            self.shape.iter().product(),
            include_str!("shaders/add.wgsl"),
        )
        .await;
        Tensor {
            buffer: result_buffer,
            size: self.size,
            shape: self.shape.clone(),
            context: self.context,
        }
    }

    fn make_staging_buffer(&self) -> Buffer {
        let size = (self.size * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let staging_buffer = self.context.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.context.queue().compute(self.context.device(), |encoder| {
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, size);
        });
        staging_buffer
    }
}

impl<'a> Add for Tensor<'a> {
    type Output = Tensor<'a>;

    fn add(self, other: Tensor<'a>) -> Tensor<'a> {
        self.add(other).block_on()
    }
}
