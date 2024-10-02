use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("WGPU backend error: {0}")]
    WGPUBackendError(#[source] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, self::Error>;
