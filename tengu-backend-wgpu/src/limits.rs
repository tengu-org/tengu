//! Module implementing the `Limits` struct for the WGPU backend. It provides information about the
//! limitations of the WGPU backend.

use crate::Backend as WGPUBackend;

/// A struct representing the limits of the WGPU backend.
pub struct Limits {
    device_limits: wgpu::Limits,
}

impl Limits {
    /// Creates a new `Limits` instance.
    ///
    /// # Parameters
    /// - `backend`: A reference to the WGPU backend.
    ///
    /// # Returns
    /// A new `Limits` instance.
    pub fn new(backend: &WGPUBackend) -> Self {
        Self {
            device_limits: backend.device().limits(),
        }
    }
}

impl tengu_backend::Limits for Limits {
    type Backend = WGPUBackend;

    /// Returns the maximum number of tensors that can be used in a single compute stage.
    /// This number is equal to the maximum buffer count for one shader stage in WGPU.
    ///
    /// # Returns
    /// The maximum number of tensors that can be used in a single compute stage. If the limit
    /// is not defined (for example in case of a CPU backend), `None` is returned.
    fn max_tensor_per_compute(&self) -> Option<usize> {
        Some(self.device_limits.max_storage_buffers_per_shader_stage as usize)
    }
}

#[cfg(test)]
mod tests {
    use crate::Backend as WGPUBackend;
    use pretty_assertions::assert_eq;
    use tengu_backend::{Backend, Limits};
    use tengu_wgpu::WGPU;
    use wgpu::Backends;

    #[tokio::test]
    async fn device_limits() {
        let max_storage_buffers_per_shader_stage = 4;
        let device = WGPU::builder()
            .backends(Backends::PRIMARY)
            .build()
            .adapter()
            .request()
            .await
            .unwrap()
            .device()
            .with_limits(wgpu::Limits {
                max_storage_buffers_per_shader_stage,
                ..Default::default()
            })
            .request()
            .await
            .unwrap();
        let backend = WGPUBackend::from_device(device);
        assert_eq!(
            backend.limits().max_tensor_per_compute(),
            Some(max_storage_buffers_per_shader_stage as usize),
        );
    }
}
