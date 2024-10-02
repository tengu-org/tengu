#![allow(async_fn_in_trait)]

use crate::{IOType, Result};

pub trait Probe<T: IOType> {
    /// Retrieve the data from the probe.
    async fn retrieve_to(&self, buffer: &mut Vec<T>) -> Result<()>;
}
