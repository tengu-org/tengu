use tengu_wgpu::{Buffer, Encoder};

use crate::probe::Probe;

mod wgpu;

#[allow(dead_code)]
pub trait RawTensor {
    type Output;

    // Unary functions.
    fn exp(&self) -> Self;
    fn log(&self) -> Self;

    // Arithematic operations.
    fn add(&self, other: &Self) -> Self;
    fn sub(&self, other: &Self) -> Self;
    fn mul(&self, other: &Self) -> Self;
    fn div(&self, other: &Self) -> Self;

    // Relational operations.
    fn eq(&self, other: &Self) -> Self;
    fn neq(&self, other: &Self) -> Self;
}

#[allow(dead_code)]
pub trait Backend {
    fn compute(&self);
}

pub trait Emit {
    fn emit(&self) -> String;
}

pub trait Datum {
    fn buffer(&self) -> &Buffer;
    fn probe(&self) -> &Probe;
    fn read(&self, encoder: &mut Encoder) {
        encoder.copy_buffer(self.buffer(), self.probe().buffer());
    }
}
