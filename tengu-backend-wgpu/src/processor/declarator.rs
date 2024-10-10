use std::collections::HashMap;

use itertools::Itertools;
use tengu_backend::StorageType;
use tengu_wgpu::BufferUsage;

use crate::source::Source;
use crate::tensor::Tensor;

const GROUP: usize = 0;

pub struct Declarator<'a> {
    declarations: HashMap<&'a str, String>,
}

impl Declarator<'_> {
    pub fn new() -> Self {
        Self {
            declarations: HashMap::new(),
        }
    }

    pub fn header(&self) -> String {
        self.declarations.values().join("\n")
    }
}

impl<'a> Declarator<'a> {
    pub fn var<T: StorageType>(&mut self, binding: usize, tensor: &'a Tensor<T>) {
        self.declarations
            .entry(tensor.label())
            .or_insert_with(|| declaration(binding, tensor));
    }
}

fn declaration<T: StorageType>(binding: usize, tensor: &Tensor<T>) -> String {
    let label = tensor.label();
    let access = access(tensor.buffer().usage());
    let ty = std::any::type_name::<T>();
    format!("@group({GROUP}) @binding({binding}) var<storage, {access}> {label}: array<{ty}>;")
}

fn access(usage: BufferUsage) -> &'static str {
    match usage {
        BufferUsage::Read => "read",
        BufferUsage::Write => "write",
        BufferUsage::ReadWrite => "read_write",
        BufferUsage::Staging => panic!("cannot declare a staging buffer in a shader"),
    }
}

// Default implementation

impl Default for Declarator<'_> {
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
        let a = backend.tensor("a", &[1.0, 2.0, 3.0, 4.0]);
        let b = backend.tensor("b", &[5.0, 6.0, 7.0, 8.0]);
        let c = backend.zero::<f32>("c", 4);
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
