//! This module provides the implementation of the `Tensor` struct, which represents a tensor on the CPU backend.
//! It includes functionality for creating tensors, and for copying and extracting their data.

use std::borrow::Cow;
use std::cell::RefCell;

use tengu_tensor::StorageType;
use tengu_tensor::Tensor as RawTensor;
use tengu_utils::Label;

mod arithmetic;
mod cast;
mod relational;
mod unary_fn;

/// Represents a tensor on the CPU backend.
pub struct Tensor<T> {
    label: Label,
    count: usize,
    shape: Vec<usize>,
    data: RefCell<Vec<T>>,
}

impl<T: StorageType> Tensor<T> {
    /// Creates a new `Tensor` with the specified backend, label, shape, and data.
    ///
    /// # Parameters
    /// - `backend`: A reference-counted pointer to the WGPU backend.
    /// - `label`: A string label for identifying the tensor.
    /// - `shape`: The shape of the tensor as a vector of unsigned integers.
    /// - `data`: The data to be stored in the tensor.
    ///
    /// # Returns
    /// A new instance of `Tensor`.
    pub fn new(label: impl Into<Label>, shape: impl Into<Vec<usize>>, data: impl Into<Vec<T>>) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            label: label.into(),
            count,
            shape,
            data: data.into().into(),
        }
    }

    /// Creates a new `Tensor` by repeating a single element across a specified shape.
    ///
    /// # Parameters
    /// - `label`: A string label for identifying the tensor.
    /// - `shape`: The shape of the tensor as a vector of unsigned integers.
    /// - `elem`: The element to be repeated.
    ///
    /// # Returns
    /// A new instance of `Tensor` holding `elem` in every cell.
    pub fn repeat(label: impl Into<Label>, shape: impl Into<Vec<usize>>, elem: T) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            label: label.into(),
            count,
            shape,
            data: vec![elem; count].into(),
        }
    }

    /// Creates a new `Tensor` with the specified shape and zero-initialized data.
    ///
    /// # Parameters
    /// - `label`: A string label for identifying the tensor.
    /// - `shape`: The shape of the tensor as a vector of unsigned integers.
    ///
    /// # Returns
    /// A new instance of `Tensor` with zero-initialized data.
    pub fn empty(label: impl Into<Label>, shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            label: label.into(),
            count,
            shape,
            data: vec![T::default(); count].into(),
        }
    }

    /// Copies data from another tensor into this tensor.
    ///
    /// # Parameters
    /// - `other`: The tensor to copy data from.
    pub fn copy_from(&self, other: &Self) {
        self.data.borrow_mut().copy_from_slice(&other.data.borrow());
    }
}

// NOTE: Backend tensor trait implementation.

impl<T: StorageType> RawTensor<T> for Tensor<T> {
    /// Returns the label of the tensor.
    ///
    /// # Returns
    /// The label of the tensor.
    fn label(&self) -> &str {
        self.label.value()
    }

    /// Returns the number of elements in the tensor.
    ///
    /// # Returns
    /// The number of elements in the tensor.
    fn count(&self) -> usize {
        self.count
    }

    /// Returns the shape of the tensor.
    ///
    /// # Returns
    /// The shape of the tensor as a slice of unsigned integers.
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Retrieves the data from the tensor.
    ///
    /// # Returns
    /// A `Cow` containing either a reference or owned buffer with the tensor data.
    async fn retrieve(&self) -> anyhow::Result<Cow<'_, [T::IOType]>> {
        let data = self
            .data
            .borrow()
            .iter()
            .map(|v| v.convert())
            .collect::<Vec<_>>()
            .into();
        Ok(data)
    }
}

// NOTE: Clone implementation.

impl<T: StorageType> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            label: self.label.clone(),
            count: self.count,
            shape: self.shape.clone(),
            data: self.data.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use std::collections::HashSet;
    use std::rc::Rc;

    use tengu_backend::{Backend, Processor};
    use tengu_tensor::{Function, Operator, Type};

    use crate::Backend as CPUBackend;

    #[test]
    fn scalar() {
        let probes = HashSet::new();
        let mut processor = CPUBackend.processor(&probes);
        let scalar = processor.scalar(2.37);
        let scalar = scalar.as_ref::<f32>();
        assert_eq!(scalar.shape, [1]);
        assert_eq!(scalar.data.borrow().len(), 1);
        assert_eq!(scalar.data.borrow().to_vec(), [2.37]);
    }

    #[test]
    fn cast() {
        let probes = HashSet::new();
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.processor(&probes);
        let a = backend.tensor("a", [2, 2], &[1, 2, 3, 4]);
        let a = processor.var(&a);
        let cast_a = processor.cast(a, Type::F32);
        let cast_a = cast_a.as_ref::<f32>();
        assert_eq!(cast_a.shape, [2, 2]);
        assert_eq!(cast_a.data.borrow().len(), 4);
        assert_eq!(cast_a.data.borrow().to_vec(), [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn unary_fn() {
        let probes = HashSet::new();
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.processor(&probes);
        let a = backend.tensor("a", [1], &[1.0]);
        let a = processor.var(&a);
        let exp = processor.unary_fn(a, Function::Exp);
        let cast_a = exp.as_ref::<f32>();
        assert_eq!(cast_a.shape, [1]);
        assert_eq!(cast_a.data.borrow().len(), 1);
        assert_eq!(cast_a.data.borrow().to_vec(), [std::f32::consts::E]);
    }

    #[test]
    fn binary() {
        let probes = HashSet::new();
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.processor(&probes);
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let b = backend.tensor("b", [4], &[5, 6, 7, 8]);
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.binary(a, b, Operator::Mul);
        println!("a_add_b is {}", a_add_b.variant());
        let a_add_b = a_add_b.as_ref::<i32>();
        assert_eq!(a_add_b.shape, [4]);
        assert_eq!(a_add_b.data.borrow().len(), 4);
        assert_eq!(a_add_b.data.borrow().to_vec(), [5, 12, 21, 32]);
    }

    #[test]
    fn statement() {
        let probes = HashSet::new();
        let backend = Rc::new(CPUBackend);
        let mut processor = backend.processor(&probes);
        let a = backend.tensor("a", [4], &[1, 2, 3, 4]);
        let b = backend.tensor("b", [4], &[5, 6, 7, 8]);
        let c = backend.zero::<i32>("c", [4]);
        let a = processor.var(&a);
        let b = processor.var(&b);
        let a_add_b = processor.binary(a, b, Operator::Add);
        let c = processor.var(&c);
        let statement = processor.statement(c, a_add_b);
        let statement = statement.as_ref::<i32>();
        assert_eq!(statement.shape, [4]);
        assert_eq!(statement.data.borrow().len(), 4);
        assert_eq!(statement.data.borrow().to_vec(), [6, 8, 10, 12]);
    }
}
