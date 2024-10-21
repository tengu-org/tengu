readme:
    cargo readme -r tengu-wgpu -o README.md

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

mnist level="error":
    RUST_LOG="none,tengu_wgpu={{level}},tengu_backend_wgpu={{level}}" cargo run --example mnist
