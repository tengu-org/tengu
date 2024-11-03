/// A enum representing the kind of the tensor. Using this information let us apply some
/// optimizations on the WGPU backend.
#[derive(Copy, Clone)]
pub enum Kind {
    /// Generic tensor kind, with data stored as a vector.
    Generic,
    /// A scalar which needs no associated buffers.
    Scalar,
}
