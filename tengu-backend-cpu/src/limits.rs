use crate::Backend as CPUBackend;

pub struct Limits;

impl tengu_backend::Limits for Limits {
    type Backend = CPUBackend;

    /// Returns the maximum number of tensors that can be used in a single compute stage.
    /// For CPU this values is always None because there is no (theoretical) limit to the number of
    /// tensors on CPU.
    ///
    /// # Returns
    /// None, indicating that there is no limit.
    fn max_tensor_per_compute(&self) -> Option<usize> {
        None
    }
}
