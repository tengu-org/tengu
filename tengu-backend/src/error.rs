use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("WGPU error: {0}")]
    WGPUError(#[source] anyhow::Error),
    #[error("Backend error: {0}")]
    BackendError(#[source] anyhow::Error),
    #[error("Storage buffer limit reached: {0} buffers used")]
    BufferLimitReached(usize),
}

pub type Result<T> = std::result::Result<T, self::Error>;
