use crate::source::Source;

pub struct Block<'a> {
    sources: Vec<&'a dyn Source>,
}

impl<'a> Block<'a> {
    pub fn new(sources: impl IntoIterator<Item = &'a dyn Source>) -> Self {
        Self {
            sources: sources.into_iter().collect(),
        }
    }

    pub fn sources(&'a self) -> impl Iterator<Item = &'a dyn Source> {
        self.sources.iter().copied()
    }
}
