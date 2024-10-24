pub struct Limits;

impl tengu_backend::Limits for Limits {
    fn max_tensor_per_compute(&self) -> Option<usize> {
        None
    }
}
