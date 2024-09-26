use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("WGPU error: {0}")]
    WGPUError(#[from] tengu_wgpu::Error),
    #[error("Internal error: {0}")]
    InternalError(#[source] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, self::Error>;
