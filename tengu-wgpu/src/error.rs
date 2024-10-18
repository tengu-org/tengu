use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Readout error: {0}")]
    ReadoutError(#[source] anyhow::Error),
    #[error("Compute error: {0}")]
    ComputeError(#[from] anyhow::Error),
    #[error("cannot create surface: {0}")]
    CreateSurfaceError(#[from] wgpu::CreateSurfaceError),
    #[error("no suitable adapter found")]
    CreateAdapterError,
    #[error("cannot create device: {0}")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
}

pub type Result<T> = std::result::Result<T, self::Error>;
