use crate::Backend;

pub struct Limits;

impl tengu_backend::Limits for Limits {
    type Backend = Backend;

    fn max_tensor_per_compute(&self) -> Option<usize> {
        None
    }
}
