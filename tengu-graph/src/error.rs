use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Tensor error: {0}")]
    TensorError(#[source] tengu_graph_tensor::Error),
    #[error("Backend error: {0}")]
    BackendError(#[from] tengu_backend::Error),
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
    #[error("Invalid method paramter: {0}")]
    ParameterError(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, self::Error>;
