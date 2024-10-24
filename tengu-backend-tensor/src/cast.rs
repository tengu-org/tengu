/// This module contains the `Type` enumeration, which represents the possible types to which a
/// cast can be done. This is used exclusively by the cast operations on tensors.

/// The type to which a cast is done. This is used exclusvely by the cast operations on tensors.
/// This enumeration corresponds to `StorageType` implementors but is semantically different, as it
/// represents possible cast operations on the backend, not possible types to be used by frontend.
pub enum Type {
    /// Boolean type.
    Bool,
    /// u32 type.
    U32,
    /// i32 type.
    I32,
    /// f32 type.
    F32,
}
