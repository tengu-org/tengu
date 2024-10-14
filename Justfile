check:
    cargo check --all-features
    cargo clippy --all-features
    cargo check --tests --all-features
    cargo clippy --tests --all-features
    cargo check --examples --all-features
    cargo clippy --examples --all-features

docs:
    cargo test --doc

test: docs
    cargo nextest run

