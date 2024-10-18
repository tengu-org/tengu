use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Retrieve error: {0}")]
    RetrieveError(#[source] anyhow::Error),
    #[error("Compute error: {0}")]
    ComputeError(#[source] anyhow::Error),
    #[error("WGPU error: {0}")]
    WGPUError(#[source] anyhow::Error),
    #[error("OS error: {0}")]
    OSError(#[source] anyhow::Error),
    #[error("Storage buffer limit reached: {0} buffers used")]
    BufferLimitReached(usize),
}

pub type Result<T> = std::result::Result<T, self::Error>;
