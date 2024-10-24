use std::borrow::Cow;
use std::cell::RefCell;

use tengu_backend_tensor::StorageType;

mod arithmetic;
mod cast;
mod relational;
mod unary_fn;

pub struct Tensor<T> {
    label: String,
    count: usize,
    shape: Vec<usize>,
    data: RefCell<Vec<T>>,
}

impl<T: StorageType> Tensor<T> {
    pub fn new(label: impl Into<String>, shape: impl Into<Vec<usize>>, data: impl Into<Vec<T>>) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            label: label.into(),
            count,
            shape,
            data: data.into().into(),
        }
    }

    pub fn repeat(label: impl Into<String>, shape: impl Into<Vec<usize>>, elem: T) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            label: label.into(),
            count,
            shape,
            data: vec![elem; count].into(),
        }
    }

    pub fn empty(label: impl Into<String>, shape: impl Into<Vec<usize>>) -> Self {
        let shape = shape.into();
        let count = shape.iter().product();
        Self {
            label: label.into(),
            count,
            shape,
            data: vec![T::default(); count].into(),
        }
    }

    pub fn copy_to(&self, other: &Self) {
        other.data.borrow_mut().copy_from_slice(&self.data.borrow());
    }
}

// NOTE: Backend tensor trait implementation.

impl<T: StorageType> tengu_backend_tensor::Tensor for Tensor<T> {
    type Elem = T;

    fn label(&self) -> &str {
        &self.label
    }

    fn count(&self) -> usize {
        self.count
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    async fn retrieve(&self) -> anyhow::Result<Cow<'_, [<Self::Elem as StorageType>::IOType]>> {
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
    use tengu_backend_tensor::{Function, Operator, Type};

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
