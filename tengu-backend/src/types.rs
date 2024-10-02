use num::Zero;
use std::fmt::Display;

/// A type that can be used to transfer data between CPU and GPU.
pub trait IOType: StorageType + bytemuck::Pod + Zero {}

impl IOType for f32 {}

impl IOType for u32 {}

impl IOType for i32 {}

/// A type that can be stored on the GPU.
pub trait StorageType: Display + Copy + Clone + 'static {
    /// The type that will be used to extract this storage type to CPU.
    type IOType: IOType;
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
