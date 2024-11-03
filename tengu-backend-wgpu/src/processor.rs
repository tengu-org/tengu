pub mod compute;
pub mod propagate;
pub mod readout;

use std::borrow::Cow;

use tengu_tensor_wgpu::Tensor;

pub type Atom<'a, T> = Option<Cow<'a, Tensor<T>>>;
