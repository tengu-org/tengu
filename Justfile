check:
    cargo check
    cargo clippy
    cargo check --tests
    cargo clippy --tests
    cargo check --examples
    cargo clippy --examples

test: check
    cargo nextest run
