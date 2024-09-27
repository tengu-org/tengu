use std::any::Any;

use tengu_wgpu::Buffer;

pub trait Describe {
    fn descriptor(&self) -> Descriptor;
    fn as_any(&mut self) -> &mut dyn Any;
}

pub struct Descriptor<'a> {
    pub count: usize,
    pub rep: String,
    pub buffers: Vec<&'a Buffer>,
}
