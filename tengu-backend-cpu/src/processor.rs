use std::borrow::Cow;

use tengu_tensor_cpu::Tensor;

pub mod compute;
pub mod propagate;
pub mod readout;

type Atom<'a, T> = Option<Cow<'a, Tensor<T>>>;
