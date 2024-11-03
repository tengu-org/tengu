# Generate readme files.
readme project:
    cargo readme -r {{project}} -o README.md

# Check Rust code for errors using nightly features.
lint:
    cargo fmt --all --check
    cargo check --all-targets --all-features
    cargo clippy -- -Dclippy::all -Wclippy::pedantic --deny "warnings"
    .rocks/bin/luacheck app/ test/

# Check Rust code for errors using nightly features.
lint-nightly:
    #!/usr/bin/env bash
    set -euxo pipefail
    config=$(paste -sd, tpcache/rustfmt-nightly.toml | tr -s ' = ' '=' | tr -d '"')
    cargo +nightly fmt --all --check -- --config $config
    cargo +nightly check --all-targets --all-features
    cargo +nightly clippy -- -Dclippy::all -Wclippy::pedantic --deny "warnings"

# Format Rust code using nightly rustfmt features.
format-nightly:
    #!/usr/bin/env bash
    set -euxo pipefail
    config=$(paste -sd, tpcache/rustfmt-nightly.toml | tr -s ' = ' '=' | tr -d '"')
    cargo +nightly fmt --all -- --config $config

# Generate documentation.
docs:
    cargo doc --no-deps --document-private-items --open

# Test the project.
test:
    cargo test --doc
    cargo nextest run

# Run mnist example.
mnist level="error":
    RUST_LOG="none,tengu_wgpu={{level}},tengu_backend_wgpu={{level}}" cargo run --example mnist
