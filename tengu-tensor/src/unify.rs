use itertools::{EitherOrBoth, Itertools};

pub trait Unify {
    type Output;
    fn dimensions(&self) -> &[usize];
    fn unify(self, other: Self) -> Option<Self::Output>
    where
        Self: Sized;
}

impl Unify for &[usize] {
    type Output = Vec<usize>;

    fn dimensions(&self) -> &[usize] {
        self
    }

    fn unify(self, other: Self) -> Option<Self::Output> {
        let shape = self
            .dimensions()
            .iter()
            .rev()
            .zip_longest(other.dimensions().iter().rev())
            .map(|v| match v {
                EitherOrBoth::Both(v1, v2) if *v1 == 1 => Some(*v2),
                EitherOrBoth::Both(v1, v2) if *v2 == 1 => Some(*v1),
                EitherOrBoth::Both(v1, v2) if v1 == v2 => Some(*v1),
                EitherOrBoth::Both(_, _) => None,
                EitherOrBoth::Left(v1) => Some(*v1),
                EitherOrBoth::Right(v2) => Some(*v2),
            })
            .rev()
            .collect::<Option<Vec<_>>>()?;
        Some(shape)
    }
}
