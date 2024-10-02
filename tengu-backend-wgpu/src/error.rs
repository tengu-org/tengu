use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("WGPU error: {0}")]
    WGPUError(#[source] anyhow::Error),
}

impl Error {
    pub fn wgpu_error(source: anyhow::Error) -> tengu_backend::Error {
        let error = Error::WGPUError(source);
        tengu_backend::Error::WGPUBackendError(error.into())
    }
}

pub type Result<T> = std::result::Result<T, self::Error>;
