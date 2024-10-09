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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn unify_same_length_fails() {
        let a = &[1, 2, 3, 6];
        let b = &[3, 3, 3, 6];
        a.unify(b).unwrap();
    }

    #[test]
    #[should_panic]
    fn unify_different_lengths_fails() {
        let a = &[2, 4, 3, 6];
        let b = &[3, 1, 1];
        a.unify(b).unwrap();
    }

    #[test]
    fn unify_same_length() {
        let a = &[1, 2, 3, 6];
        let b = &[3, 1, 3, 1];
        let c = a.unify(b).unwrap();
        assert_eq!(c, vec![3, 2, 3, 6]);
    }

    #[test]
    fn unify_different_lengths() {
        let a = &[2, 4, 1, 6];
        let b = &[1, 3, 1];
        let c = a.unify(b).unwrap();
        assert_eq!(c, vec![2, 4, 3, 6]);
    }
}
