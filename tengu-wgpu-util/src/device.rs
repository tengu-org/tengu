use std::ops::Deref;

pub struct Device {
    adapter: wgpu::Adapter,
    device: wgpu::Device,
}

impl Device {
    pub fn new(adapter: wgpu::Adapter, device: wgpu::Device) -> Device {
        Self { adapter, device }
    }

    pub fn parent_adapter(&self) -> &wgpu::Adapter {
        &self.adapter
    }
}

impl Deref for Device {
    type Target = wgpu::Device;
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}
