use indoc::formatdoc;
use itertools::Itertools;
use tracing::trace;

use tengu_backend::{Error, Result};
use tengu_wgpu::Pipeline;

use crate::operation::compute::IR;

use super::{statement::Statement, Processor};

pub struct Block {
    expression: String,
    count: usize,
    declarations: Vec<String>,
}

impl Block {
    pub fn new(declarations: impl IntoIterator<Item = String>, exprs: impl IntoIterator<Item = Statement>) -> Self {
        let statements = exprs.into_iter().collect::<Vec<_>>();
        let count = statements
            .iter()
            .map(|stmt| stmt.count())
            .max()
            .expect("should have at least one statement to make a block");
        let expression = statements.iter().map(|stmt| stmt.expression()).join("    \n");
        Self {
            expression,
            count,
            declarations: declarations.into_iter().collect(),
        }
    }

    pub fn into_ir(self, label: &str, processor: &Processor) -> Result<IR> {
        let pipeline = self.pipeline(label, processor)?;
        Ok(IR::new(pipeline, self.count))
    }

    /// Returns the maximum number of elements use by the tensors in the AST.
    ///
    /// # Returns
    /// The number of elements as a `usize`.
    pub fn count(&self) -> usize {
        self.count
    }

    pub fn header(&self) -> String {
        self.declarations.join("\n")
    }

    /// Generates the body of the compute shader.
    ///
    /// # Returns
    /// A `String` containing the shader body with all expressions.
    pub fn body(&self) -> String {
        formatdoc!(
            r"
            @compute
            @workgroup_size(64)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {{
                let idx = global_id.x;
                {}
            }}",
            self.expression,
        )
    }

    /// Returns the generated shader code as a string slice.
    ///
    /// # Returns
    /// A string slice with the final shader code.
    pub fn shader(&self) -> String {
        let header = self.header();
        let body = self.body();
        trace!("Emitting shader for a block");
        format!("{}\n\n{}", header, body)
    }

    /// Creates a pipeline for the compute operations using the given processor.
    ///
    /// # Parameters
    /// - `processor`: A reference to the `Processor` object which provides shader and buffer information.
    ///
    /// # Returns
    /// A `Result` containing the `Pipeline` object if the pipeline creation is successful, or an `Error` if
    /// the buffer limit is reached.
    fn pipeline(&self, label: &str, processor: &Processor) -> Result<Pipeline> {
        trace!("Creating pipeline");
        let shader = processor.device.shader(label, &self.shader());
        let buffers = processor.sources().map(|source| source.buffer()).collect::<Vec<_>>();
        let max_buffers = processor.device.limits().max_storage_buffers_per_shader_stage as usize;
        trace!("Max buffer limit: {max_buffers}");
        if buffers.len() > max_buffers {
            return Err(Error::BufferLimitReached(max_buffers));
        }
        let pipeline = processor
            .device
            .layout()
            .add_entries(buffers)
            .pipeline(label)
            .build(shader);
        Ok(pipeline)
    }
}
