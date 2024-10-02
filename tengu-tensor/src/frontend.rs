mod binary;
mod cast;
mod expression;
mod ops;
mod unary_fn;

pub use expression::Expression;

use crate::{
    backend::{Datum, Emit},
    visitor::Visitor,
};

pub trait Shape {
    fn shape(&self) -> &[usize];
    fn count(&self) -> usize;
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
