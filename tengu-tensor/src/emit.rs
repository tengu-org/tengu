use downcast_rs::{impl_downcast, Downcast};

pub trait Emit: Downcast {
    fn emit(&self) -> String;
}

impl_downcast!(Emit);
