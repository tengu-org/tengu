use super::Source;

impl<'a> Source<'a> {
    pub fn copy_from(&self, other: &Self) {
        match (self, other) {
            (Source::U32(_), Source::U32(_)) => other.as_ref::<u32>().copy_to(self.as_ref::<u32>()),
            (Source::I32(_), Source::I32(_)) => other.as_ref::<i32>().copy_to(self.as_ref::<i32>()),
            (Source::F32(_), Source::F32(_)) => other.as_ref::<f32>().copy_to(self.as_ref::<f32>()),
            (lhs, rhs) => panic!("Cannot copy from {} to {}", rhs.variant(), lhs.variant()),
        }
    }
}
