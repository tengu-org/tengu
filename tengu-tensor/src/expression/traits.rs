use tengu_wgpu::{Buffer, Encoder};

use crate::{probe::Probe, visitor::Visitor};

pub trait Shape {
    fn shape(&self) -> &[usize];
    fn count(&self) -> usize;
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

pub trait Source: Shape + Emit + Datum {
    fn label(&self) -> &str;
    fn declaration(&self, group: usize, binding: usize) -> String;
}

pub trait Node: Shape + Emit {
    fn visit<'a>(&'a self, visitor: &mut Visitor<'a>);
    fn clone_box(&self) -> Box<dyn Node>;
    fn source(&self) -> Option<&dyn Source>;
}
