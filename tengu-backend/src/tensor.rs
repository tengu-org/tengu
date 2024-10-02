use crate::{Probe, StorageType};

pub trait Tensor<T: StorageType> {
    /// The type of probe this tensor can be bound to.
    type Probe: Probe<T::IOType>;

    /// Return the label of the tensor.
    fn label(&self) -> &str;

    /// Gets tensor's probe or inits a new one and bounds it to this tensor.
    fn probe(&self) -> &Self::Probe;
}
