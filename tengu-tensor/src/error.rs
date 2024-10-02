use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("WGPU error: {0}")]
    BackendError(#[from] tengu_backend::Error),
    #[error("Internal error: {0}")]
    InternalError(#[source] anyhow::Error),
    #[error("Cannot find source with label {0}")]
    SourceNotFound(String),
    #[error("Link path {0} does not contain '/'")]
    InvalidLinkPath(String),
    #[error("Cannot find block with label {0}")]
    BlockNotFound(String),
    #[error("Block with id {0} already exists in the graph")]
    BlockAlreadyExists(String),
    #[error("Types don't match")]
    TypeMismatch,
    #[error("Shapes don't match")]
    ShapeMismatch,
}

pub type Result<T> = std::result::Result<T, self::Error>;
