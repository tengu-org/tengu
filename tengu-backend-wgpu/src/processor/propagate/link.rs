use tengu_wgpu::{Buffer, Encoder};

pub struct Link<'a> {
    from: &'a Buffer,
    to: &'a Buffer,
}

impl<'a> Link<'a> {
    pub fn new(from: &'a Buffer, to: &'a Buffer) -> Self {
        Self { from, to }
    }

    pub fn copy(&self, encoder: &mut Encoder) {
        encoder.copy_buffer(self.from, self.to);
    }
}
