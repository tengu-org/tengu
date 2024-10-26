//! This module provides various tensor-related utilities to help with implementation.

use random_string::charsets::ALPHA;

/// The length of the label generated for tensors if no label is provided.
const LABEL_LENGTH: usize = 6;

/// Creates a new random label for the tensor.
///
/// # Returns
/// A string representing the label of the tensor.
pub fn create_label() -> String {
    random_string::generate(LABEL_LENGTH, ALPHA)
}
