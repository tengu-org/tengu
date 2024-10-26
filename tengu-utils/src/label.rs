use std::fmt::Display;

use random_string::charsets::ALPHA;

/// The length of the label generated for tensors if no label is provided.
const LABEL_LENGTH: usize = 6;

#[derive(Clone, Debug)]
pub struct Label {
    label: String,
}

impl Label {
    pub fn new() -> Self {
        let label = random_string::generate(LABEL_LENGTH, ALPHA);
        Self { label }
    }

    pub fn value(&self) -> &str {
        &self.label
    }
}

impl Default for Label {
    fn default() -> Self {
        Self::new()
    }
}

impl Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

impl AsRef<str> for Label {
    fn as_ref(&self) -> &str {
        &self.label
    }
}

// NOTE: From traits.

impl<'a> From<&'a str> for Label {
    fn from(label: &'a str) -> Self {
        Self { label: label.into() }
    }
}

impl From<String> for Label {
    fn from(label: String) -> Self {
        Self { label }
    }
}
