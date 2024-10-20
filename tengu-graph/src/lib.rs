//! The `tengu-tensor` crate provides a comprehensive framework for tensor operations, supporting
//! various computational backends and offering a rich set of functionalities for building, manipulating,
//! and evaluating tensor expressions. It is designed to be extensible, efficient, and user-friendly,
//! making it suitable for both research and production environments.
//!
//! # Design Philosophy
//!
//! The crate is designed with the following principles in mind:
//!
//! - **Modularity**: The library is composed of several well-defined modules, each responsible for specific
//!   aspects of tensor computation. This modularity makes it easy to extend and maintain.
//! - **Backend Agnostic**: By abstracting over computational backends, the crate allows users to switch
//!   between different execution environments (e.g., CPU, GPU) with minimal code changes.
//! - **Ease of Use**: With a clear and concise API, the library aims to provide an intuitive experience
//!   for users, enabling them to focus on their computational tasks without worrying about low-level details.
//! - **Performance**: Leveraging Rust's performance characteristics and efficient backend implementations,
//!   the crate aims to deliver high-performance tensor computations suitable for both small and large-scale tasks.
//!
//! # Key Modules
//!
//! - `error`: Defines error handling mechanisms and result types used throughout the crate.
//! - `expression`: Contains the core structures and operations for tensor expressions, including
//!   unary and binary operations, as well as relational and arithmetic operations.
//! - `graph`: Manages computation graphs, enabling efficient execution of tensor operations.
//! - `probe`: Includes tools for inspecting and debugging tensor expressions and computation graphs.
//! - `tengu`: The main entry point of the crate, providing high-level functionalities and integration
//!   with different backends.
//! - `tensor`: Defines tensor structures and associated methods for manipulation and computation.
//!
//! # Usage
//!
//! To use the `tengu-tensor` crate, you need to include it as a dependency in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! tengu-tensor = "0.1.0"
//! ```
//!
//! Here is a basic example demonstrating how to create and manipulate tensors using the crate:
//!
//! ```rust
//! use tengu_graph::{Tengu, StorageType};
//!
//! #[tokio::main]
//! async fn main() {
//!     // Initialize the Tengu context with the WGPU backend.
//!     let tengu = Tengu::wgpu().await.unwrap();
//!
//!     // Create a tensor filled with zeros.
//!     let a = tengu.tensor([2, 3]).label("a").zero::<f32>();
//!
//!     // Create computation graph.
//!     let mut graph = tengu.graph();
//!     graph
//!         .add_block("main")
//!         .unwrap()
//!         .add_computation("addoe", a + 1.0);
//!
//!     // Set up probe.
//!     let mut add = graph.add_probe::<f32>("main/a").unwrap();
//!
//!     // Run the computation and display the result.
//!     graph.compute(1).await;
//!     println!("{:?}", add.retrieve().await.unwrap());
//! }
//! ```
//!
//! This example initializes the `Tengu` context with the WGPU backend, creates a tensor filled with zeros,
//! performs some arithmetic operations on it, and prints the result. The crate's API allows for concise
//! and readable code, making it easy to perform complex tensor manipulations.
//!
//! # Error Handling
//!
//! The crate uses the `Result` and `Error` types defined in the `error` module for error handling. These
//! types are re-exported at the crate level for convenience. Errors can occur during tensor operations,
//! shape unifications, and backend initializations, and the error types provide detailed information to
//! help diagnose and resolve issues.

mod builder;
mod channel;
mod collector;
mod error;
mod expression;
mod graph;
mod node;
mod probe;
mod source;
mod tengu;
mod tensor;
mod unify;

pub use error::{Error, Result};
pub use tengu::Tengu;
pub use tengu_tensor_traits::{IOType, StorageType};
