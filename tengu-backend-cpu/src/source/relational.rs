use super::Source;

pub trait Equality<Rhs = Self> {
    type Output;

    fn eq(&self, other: &Rhs) -> Self::Output;

    fn neq(&self, other: &Rhs) -> Self::Output;
}

impl<'a> Equality for Source<'a> {
    type Output = Self;

    fn eq(&self, other: &Self) -> Self::Output {
        match (self, other) {
            (Source::U32(_), Source::U32(_)) => (self.as_ref::<u32>().eq(other.as_ref::<u32>())).into(),
            (Source::I32(_), Source::I32(_)) => (self.as_ref::<i32>().eq(other.as_ref::<i32>())).into(),
            (Source::F32(_), Source::F32(_)) => (self.as_ref::<f32>().eq(other.as_ref::<f32>())).into(),
            (lhs, rhs) => panic!(
                "Comparison operations are not implemented for {} and {}",
                lhs.variant(),
                rhs.variant()
            ),
        }
    }

    fn neq(&self, other: &Self) -> Self::Output {
        match (self, other) {
            (Source::U32(_), Source::U32(_)) => (self.as_ref::<u32>().neq(other.as_ref::<u32>())).into(),
            (Source::I32(_), Source::I32(_)) => (self.as_ref::<i32>().neq(other.as_ref::<i32>())).into(),
            (Source::F32(_), Source::F32(_)) => (self.as_ref::<f32>().neq(other.as_ref::<f32>())).into(),
            (lhs, rhs) => panic!(
                "Comparison operations are not implemented for {} and {}",
                lhs.variant(),
                rhs.variant()
            ),
        }
    }
}
