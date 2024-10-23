use super::{Borrowed, Owned, Source};

impl<'a> Source<'a> {
    pub fn copy_from(&self, rhs: &Self) {
        match (self, rhs) {
            (Source::Owned(lhs), Source::Owned(rhs)) => lhs.copy_from(rhs),
            (Source::Owned(lhs), Source::Borrowed(rhs)) => lhs.copy_from(rhs),
            (Source::Borrowed(lhs), Source::Owned(rhs)) => lhs.copy_from(rhs),
            (Source::Borrowed(lhs), Source::Borrowed(rhs)) => lhs.copy_from(rhs),
        }
    }
}

trait CopyFrom<Rhs = Self> {
    fn copy_from(&self, other: &Rhs);
}

macro_rules! impl_copy_from {
    ( $lhs:ident, $rhs:ident ) => {
        fn copy_from(&self, other: &$rhs) {
            match (self, other) {
                ($lhs::Bool(lhs), $rhs::Bool(rhs)) => rhs.copy_to(lhs),
                ($lhs::U32(lhs), $rhs::U32(rhs)) => rhs.copy_to(lhs),
                ($lhs::I32(lhs), $rhs::I32(rhs)) => rhs.copy_to(lhs),
                ($lhs::F32(lhs), $rhs::F32(rhs)) => rhs.copy_to(lhs),
                (lhs, rhs) => panic!("Cannot copy from {} to {}", rhs.variant(), lhs.variant()),
            }
        }
    };
}

impl<'a> CopyFrom for Borrowed<'a> {
    impl_copy_from!(Borrowed, Borrowed);
}

impl<'a> CopyFrom<Owned> for Borrowed<'a> {
    impl_copy_from!(Borrowed, Owned);
}

impl<'a> CopyFrom<Borrowed<'a>> for Owned {
    impl_copy_from!(Owned, Borrowed);
}

impl CopyFrom<Owned> for Owned {
    impl_copy_from!(Owned, Owned);
}
