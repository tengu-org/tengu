use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Compute error")]
    ComputeError(#[from] anyhow::Error),
    #[error("cannot create surface: {0}")]
    CreateSurfaceError(#[from] wgpu::CreateSurfaceError),
    #[error("no suitable adapter found")]
    CreateAdapterError,
    #[error("cannot create device: {0}")]
    RequestDeviceError(#[from] wgpu::RequestDeviceError),
}

pub type Result<T> = std::result::Result<T, self::Error>;
