//! This module defines the `Declarator` struct and associated functions for managing
//! shader variable declarations in a WGPU-based tensor computation framework.
//!
//! The `Declarator` struct is used to create and manage storage variable declarations
//! for tensors, which are then used in shader programs.

use itertools::Itertools;
use std::collections::HashMap;
use tengu_tensor::StorageType;
use tengu_wgpu::BufferUsage;

use crate::source::Source;
use crate::tensor::Tensor;

const GROUP: usize = 0;

/// A struct for declaring shader storage variables.
pub struct Declarator<'a> {
    declarations: HashMap<&'a str, String>,
}

impl Declarator<'_> {
    /// Creates a new `Declarator`.
    ///
    /// # Returns
    /// A new instance of `Declarator`.
    pub fn new() -> Self {
        Self {
            declarations: HashMap::new(),
        }
    }

    /// Generates a header string containing all the declarations.
    ///
    /// # Returns
    /// A `String` containing the header with all declarations.
    pub fn header(&self) -> String {
        self.declarations.values().join("\n")
    }
}

// NOTE: Processing interface

impl<'a> Declarator<'a> {
    /// Processes a variable by adding it to the list of tensor declarations.
    /// not there yet.
    ///
    /// # Parameters
    /// - `binding`: The binding index for the shader variable.
    /// - `tensor`: The tensor to declare as a shader variable.
    pub fn var<T: StorageType>(&mut self, binding: usize, tensor: &'a Tensor<T>) {
        self.declarations
            .entry(tensor.label().expect("input tensors should have a label"))
            .or_insert_with(|| declaration(binding, tensor));
    }
}

/// Generates a declaration string for a tensor.
///
/// # Parameters
/// - `binding`: The binding index for the shader variable.
/// - `tensor`: The tensor to declare as a shader variable.
///
/// # Returns
/// A `String` containing the declaration for the tensor.
fn declaration<T: StorageType>(binding: usize, tensor: &Tensor<T>) -> String {
    let label = tensor.label().expect("input tensors should have a label");
    let access = access(tensor.buffer().usage());
    let ty = std::any::type_name::<T>();
    format!("@group({GROUP}) @binding({binding}) var<storage, {access}> {label}: array<{ty}>;")
}

/// Determines the access type for a buffer based on its usage.
///
/// # Parameters
/// - `usage`: The buffer usage type.
///
/// # Returns
/// A static string representing the access type.
fn access(usage: BufferUsage) -> &'static str {
    match usage {
        BufferUsage::Read => "read",
        BufferUsage::Write => "write",
        BufferUsage::ReadWrite => "read_write",
        BufferUsage::Staging => panic!("cannot declare a staging buffer in a shader"),
    }
}

// NOTE: Default implementation

impl Default for Declarator<'_> {
    /// Creates a new `Declarator` using the default implementation.
    ///
    /// # Returns
    /// A new instance of `Declarator`.
    fn default() -> Self {
        Declarator::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Backend as WGPUBackend;

    use regex::RegexSet;
    use tengu_backend::Backend;

    #[tokio::test]
    async fn declaration() {
        let backend = WGPUBackend::new().await.unwrap();
        let a = backend.tensor("a", [4], &[1.0, 2.0, 3.0, 4.0]);
        let b = backend.tensor("b", [4], &[5.0, 6.0, 7.0, 8.0]);
        let c = backend.zero::<f32>("c", [4]);
        let mut processor = Declarator::new();
        processor.var(0, &a);
        processor.var(1, &b);
        processor.var(2, &c);
        let header = processor.header();
        let declarations = header.lines().collect::<Vec<_>>();
        let re = RegexSet::new([
            r"@group\(0\) @binding\(\d+\) var<storage, read> a: array<f32>;",
            r"@group\(0\) @binding\(\d+\) var<storage, read> b: array<f32>;",
            r"@group\(0\) @binding\(\d+\) var<storage, read_write> c: array<f32>;",
        ])
        .unwrap();
        for declaration in declarations {
            println!("{:?}", declaration);
            assert!(re.is_match(declaration));
        }
    }
}
