/// This module defined the `Cage` enum that works similarly to `Cow`, but holds either a borrowed
/// reference to `dyn Any` or or boxed version of it. It is useful when you need to hold an object
/// of any type but sometimes you just need to borrow it and somethimes you want to place it on the
/// heap.
use std::any::Any;

/// The `Cage` enum holds either a borrowed reference to `dyn Any` or a boxed version of it.
pub enum Cage<'a> {
    Borrowed(&'a dyn Any),
    Owned(Box<dyn Any>),
}

impl<'a> Cage<'a> {
    /// Creates a new `Cage` instance holding an owned value.
    ///
    /// # Parameters
    /// - `value`: The value to hold.
    ///
    /// # Returns
    /// A new instance of an owned `Cage`.
    pub fn owned<T: Clone + 'static>(value: T) -> Self {
        Self::Owned(Box::new(value))
    }

    /// Creates a new `Cage` instance holding a borrowed reference.
    ///
    /// # Parameters
    /// - `value`: A reference to the value to hold.
    ///
    /// # Returns
    /// A new instance of a borrowed `Cage`.
    pub fn borrowed<T: Clone + 'static>(value: &'a T) -> Self {
        Self::Borrowed(value)
    }

    /// Converts the `Cage` into an owned inner value downcasting it to the specified type in the
    /// process. If the `Cage` was borrowed, the value will be cloned and the copy moved out.
    /// If the `Cage` was owned, the box holding the trait object will be converted into the type
    /// and the value moved out of the box. The cage is consumed in the process.
    ///
    /// # Returns
    /// The owned inner value if it can be downcasted to the specified type.
    pub fn into_owned<T: Clone + 'static>(self) -> Option<T> {
        match self {
            Self::Borrowed(tensor) => tensor.downcast_ref::<T>().cloned(),
            Self::Owned(tensor) => tensor.downcast::<T>().ok().map(|b| *b),
        }
    }

    /// Returns a reference to the inner value downcasting it to the specified type. The `Cage` is
    /// not consumed, only the reference is returned.
    ///
    /// # Returns
    /// A reference to the inner value if it can be downcasted to the specified type.
    pub fn as_ref<T: Clone + 'static>(&self) -> Option<&T> {
        match self {
            Self::Borrowed(tensor) => tensor.downcast_ref::<T>(),
            Self::Owned(tensor) => tensor.downcast_ref::<T>(),
        }
    }

    /// Clones the inner value and returns a new owned `Cage` instance holding the cloned value. The
    /// `Cage` is *not* consumed in the process.
    ///
    /// # Returns
    /// A new instance of an owned `Cage` holding the cloned value.
    pub fn cloned<T: Clone + 'static>(&self) -> Option<Self> {
        let inner = self.as_ref::<T>().cloned()?;
        Some(Cage::owned(inner))
    }

    /// Lifts the value from borrowed to owned state. The original `Cage` is consumed and the new
    /// owned version is returned. If the `Cage` was originally owned, just returns `self`.
    ///
    /// # Returns
    /// The owned version of the `Cage`.
    pub fn lift<T: Clone + 'static>(self) -> Self {
        match self {
            Self::Owned(_) => self,
            Self::Borrowed(tensor) => Self::Owned(Box::new(tensor.downcast_ref::<T>().unwrap().clone())),
        }
    }
}
