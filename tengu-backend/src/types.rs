//! This module defines traits for types that can be transferred between the CPU and GPU
//! (`IOType`) and types that can be stored on the GPU (`StorageType`). These traits ensure
//! that the types used in tensor computations are compatible with GPU operations and can
//! be safely managed.
//!
//! # Traits
//!
//! - `IOType`: A trait for types that can be used to transfer data between CPU and GPU.
//! - `StorageType`: A trait for types that can be stored on the GPU.
//!
//! # Implementors
//!
//! Various primitive types implement these traits to facilitate their use in GPU computations:
//!
//! - `f32`
//! - `u32`
//! - `i32`
//! - `bool` (only for `StorageType`, with `u32` as the associated `IOType`)

use std::fmt::{Debug, Display};

/// A type that can be used to transfer data between CPU and GPU.
///
/// This trait combines the properties of `StorageType` and `bytemuck::Pod`
/// to ensure that any type implementing `IOType` can be safely transferred and managed
/// between CPU and GPU memory.
///
/// # Implementors
/// - `f32`
/// - `u32`
/// - `i32`
///
/// # Safety
/// Implementors must ensure that the type is `Pod` (Plain Old Data) which means it
/// can be safely transmuted to and from byte arrays without any undefined behavior.
pub trait IOType: StorageType + bytemuck::Pod {}

impl IOType for f32 {}

impl IOType for u32 {}

impl IOType for i32 {}

/// A type that can be stored on the GPU.
///
/// This trait ensures that any type implementing `StorageType` can be safely copied,
/// cloned, and displayed. Additionally, it defines an associated type `IOType`
/// which specifies the type used for transferring data to the CPU. Some types, like
/// bool, can be stored on the GPU but cannot be used to download from or upload data to the GPU.
/// It is therefore necessary to distinguish between the types, and restrict certain operations
/// (like retrieving a probe) to specific subset, `IOType` or `StorageType`.
///
/// # Implementors
/// - `f32`
/// - `u32`
/// - `i32`
/// - `bool`
///
/// # Associated Types
/// - `IOType`: Defines the type that will be used to transfer this storage type to the CPU.
pub trait StorageType: Debug + Display + Copy + Clone + Default + Sized + Send + Sync + 'static
where
    Self::IOType: From<Self>,
{
    /// The type that will be used to extract this storage type to CPU.
    type IOType: IOType;

    /// Converts the storage type to the IO type.
    ///
    /// # Returns
    /// The IO type corresponding to the storage type.
    fn convert(self) -> Self::IOType {
        self.into()
    }
}

impl StorageType for f32 {
    type IOType = f32;
}

impl StorageType for u32 {
    type IOType = u32;
}

impl StorageType for i32 {
    type IOType = i32;
}

impl StorageType for bool {
    type IOType = u32;
}
