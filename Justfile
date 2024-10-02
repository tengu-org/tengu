check:
    cargo check
    cargo clippy
    cargo check --examples
    cargo clippy --examples

test: check
    cargo nextest run
