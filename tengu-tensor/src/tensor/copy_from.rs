pub trait CopyFrom {
    /// The type of the linker that performs the actual copy process.
    type Linker;

    /// Copies data from another tensor into this tensor.
    ///
    /// # Parameters
    /// - `other`: The tensor to copy data from.
    fn copy_from(&self, other: &Self, linker: &mut Self::Linker);
}
