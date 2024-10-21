//! This crate provides a high-level abstraction over the WGPU graphics API, focusing on ease of use and integration with Tengu
//! projects. It includes a set of utilities and wrappers to facilitate the creation and management of GPU resources, shaders, and
//! pipelines.
//!
//! ## Features
//!
//! - Adapter Management: Discover and initialize GPU adapters with configurable options.
//! - Device Handling: Build GPU devices, manage resources like buffers, and submit commands for execution.
//! - Pipeline Creation: Easily set up compute and render pipelines with flexible configuration.
//! - Surface Integration: Handle GPU surfaces for display output management.
//!
//! ## Modules
//!
//! - `adapter`: Manages GPU adapters and provides functionality to request adapters compatible with specific surfaces.
//! - `buffer`: Contains utilities for creating and managing GPU buffers.
//! - `device`: Wraps the WGPU device and queue, providing methods for creating encoders, buffers, and shaders.
//! - `encoder`: Manages command encoders for recording commands.
//! - `error`: Defines error types and results used across the crate.
//! - `pipeline`: Provides utilities for creating and managing GPU pipelines.
//! - `size`: Contains utilities for working with sizes and byte sizes.
//! - `surface`: Manages GPU surfaces and their configurations.
//! - `wgpu`: Contains the main entry point for creating WGPU instances and requesting default contexts.
//!
//! ## Integration with WGPU
//!
//! This crate is built on top of WGPU, providing a more ergonomic interface for common tasks while leveraging the power of WGPU's API.
//! It abstracts away much of the boilerplate code required to set up GPU resources, making it easier to create complex graphics and
//! compute applications.
//!
//! ## Example
//!
//! Below is a complete example demonstrating the workflow:
//!
//! ```rust
//! use tengu_wgpu::{WGPU, BufferUsage};
//! use wgpu::Backends;
//! use winit::event_loop::EventLoop;
//! use winit::window::Window;
//!
//! #[allow(deprecated)]
//! async fn compute_example(window: &Window) {
//!     let event_loop = EventLoop::new().unwrap();
//!     let window = event_loop.create_window(Window::default_attributes()).unwrap();
//!
//!     let wgpu_instance = WGPU::builder().backends(Backends::PRIMARY).build();
//!     let surface = wgpu_instance.create_surface(window).unwrap();
//!     let adapter = wgpu_instance.adapter().with_surface(&surface).request().await.unwrap();
//!     let device = adapter.device().request().await.unwrap();
//!     
//!     let shader_source = r#"
//!     @compute @workgroup_size(64)
//!     fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
//!         // Compute shader code
//!     }
//!     "#;
//!     let shader = device.shader("compute_shader", shader_source);
//!     let buffer = device.buffer::<u32>("buf", BufferUsage::ReadWrite).with_data(&[0; 64]);
//!     let pipeline = device.layout().add_entry(&buffer).pipeline("compute_pipeline").build(shader);
//!     let command_buffer = device
//!         .encoder("encoder")
//!         .pass("pass", |mut pass| {
//!             pass.set_pipeline(&pipeline);
//!             pass.set_bind_group(0, pipeline.bind_group(), &[]);
//!             pass.dispatch_workgroups(64, 1, 1);
//!             Ok(())
//!         })
//!         .unwrap()
//!         .finish();
//!     
//!     device.submit(command_buffer);
//! }
//! ```
//!
//! This example demonstrates how to set up a compute shader using the `tengu-wgpu` crate, providing a clear and concise workflow for
//! GPU compute operations.

mod adapter;
mod buffer;
mod device;
mod encoder;
mod error;
mod pipeline;
mod size;
mod surface;
mod wgpu;

pub use adapter::Adapter;
pub use buffer::{Buffer, BufferUsage};
pub use device::Device;
pub use encoder::Encoder;
pub use error::{Error, Result};
pub use pipeline::Pipeline;
pub use size::ByteSize;
pub use surface::BoundSurface;
pub use surface::Surface;
pub use wgpu::WGPU;
