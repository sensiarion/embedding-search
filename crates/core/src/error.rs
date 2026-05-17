#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("config: {0}")]
    Config(String),
    #[error("database: {0}")]
    Db(#[from] rusqlite::Error),
    #[error("embedding: {0}")]
    Embed(String),
    #[error("index: {0}")]
    Index(String),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("chunker: {0}")]
    Chunker(String),
}

pub type Result<T> = std::result::Result<T, Error>;
