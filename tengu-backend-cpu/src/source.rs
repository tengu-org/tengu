pub trait Source {
    fn label(&self) -> &str;

    fn readout(&self);

    fn count(&self) -> usize;
}
