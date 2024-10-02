use tengu_wgpu::{Buffer, Encoder};

pub trait Source {
    fn label(&self) -> &str;
    fn buffer(&self) -> &Buffer;
    fn readout(&self, encoder: &mut Encoder);
    fn count(&self) -> usize;
}
