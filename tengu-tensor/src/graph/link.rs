use tengu_wgpu::Encoder;

use crate::expression::traits::Source;

pub struct Link<'a> {
    from: &'a dyn Source,
    to: Vec<&'a dyn Source>,
}

impl<'a> Link<'a> {
    pub fn new(from: &'a dyn Source, to: Vec<&'a dyn Source>) -> Self {
        Self { from, to }
    }

    pub fn compute(&'a self, encoder: &mut Encoder) {
        for to in &self.to {
            encoder.copy_buffer(self.from.buffer(), to.buffer());
        }
    }
}
