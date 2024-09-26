use std::ops::Deref;
use wgpu::util::DeviceExt;

use crate::{Device, Size};

#[derive(Copy, Clone, Debug)]
pub enum BufferUsage {
    Staging,
    Read,
    ReadWrite,
    Write,
}

impl BufferUsage {
    fn usage(&self) -> wgpu::BufferUsages {
        use wgpu::BufferUsages as Usage;
        match self {
            Self::Staging => Usage::MAP_READ | Usage::COPY_DST,
            Self::Read => Usage::STORAGE | Usage::COPY_SRC,
            Self::Write => Usage::STORAGE | Usage::COPY_DST,
            Self::ReadWrite => Usage::STORAGE | Usage::COPY_SRC | Usage::COPY_DST,
        }
    }
}

pub struct Buffer {
    buffer: wgpu::Buffer,
    usage: BufferUsage,
}

impl Buffer {
    fn new(buffer: wgpu::Buffer, usage: BufferUsage) -> Self {
        Self { buffer, usage }
    }

    pub fn usage(&self) -> BufferUsage {
        self.usage
    }
}

impl Deref for Buffer {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

// BufferBuilder implementation

pub struct BufferBuilder<'a, 'device> {
    device: &'device Device,
    label: Option<&'a str>,
    usage: BufferUsage,
}

impl<'a, 'device> BufferBuilder<'a, 'device> {
    pub fn new(device: &'device Device, usage: BufferUsage) -> Self {
        Self {
            device,
            label: None,
            usage,
        }
    }

    pub fn with_label(mut self, label: &'a str) -> Self {
        self.label = Some(label);
        self
    }

    pub fn empty(self, size: Size) -> Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: self.label,
            size: size.0 as u64,
            usage: self.usage.usage(),
            mapped_at_creation: false,
        });
        Buffer::new(buffer, self.usage)
    }

    pub fn with_data<T>(self, data: &'a [T]) -> Buffer
    where
        T: bytemuck::Pod,
    {
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: self.label,
            contents: bytemuck::cast_slice(data),
            usage: self.usage.usage(),
        });
        Buffer::new(buffer, self.usage)
    }
}
