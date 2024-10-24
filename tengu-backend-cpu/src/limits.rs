//! Module implementing the `Limits` struct for the CPU backend. It provides information about the
//! limitations of the CPU backend, which is theoretically limitless in all respects.

pub struct Limits;

impl tengu_backend::Limits for Limits {
    /// Returns the maximum number of tensors that can be used in a single compute stage.
    /// In case of this CPU backend, the limit is not defined and `None` is returned.
    ///
    /// # Returns
    /// The maximum number of tensors that can be used in a single compute stage. This implementation
    /// returns `None`, signifying that there is no limit to the numer of tensor per compute pass.
    fn max_tensor_per_compute(&self) -> Option<usize> {
        None
    }
}
