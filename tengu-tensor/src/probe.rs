use num::Zero;
use tengu_backend::{Backend, IOType, StorageType, Tensor};

use crate::{Error, Result};

pub struct Probe<'a, T: StorageType, B: Backend> {
    probe: &'a <B::Tensor<T> as Tensor<T>>::Probe,
    buffer: Vec<T::IOType>,
    on: bool,
}

impl<'a, T: StorageType, B: Backend> Probe<'a, T, B>
where
    T::IOType: Zero,
{
    pub fn new(probe: &'a <B::Tensor<T> as Tensor<T>>::Probe, count: usize) -> Self {
        Self {
            probe,
            buffer: vec![T::IOType::zero(); count],
            on: true,
        }
    }

    pub fn turn_off(&mut self) {
        self.on = false;
    }

    pub fn turn_on(&mut self) {
        self.on = true;
    }

    pub async fn retrieve(&mut self) -> Result<&[T::IOType]>
    where
        T: IOType,
    {
        use tengu_backend::Probe;
        if !self.on {
            return Ok(&self.buffer);
        }
        self.probe
            .retrieve_to(&mut self.buffer)
            .await
            .map_err(Error::BackendError)?;
        Ok(&self.buffer)
    }
}
