use thiserror::Error;

#[derive(Debug, Error)]
pub enum Error {
    #[error("Channel error: {0}")]
    ChannelError(#[source] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, self::Error>;
