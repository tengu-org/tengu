pub trait Computation {
    fn emit(&self, idx: &str) -> String;
}
