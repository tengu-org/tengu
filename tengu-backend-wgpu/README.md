# tengu-backend-wgpu

This crate provides an implementation of the `tengu-backend` using WGPU, a modern graphics API for GPU-based computation.
This implementation leverages WGPU to perform efficient tensor operations, compute passes, and data management on the GPU.
The crate is structured into several modules, each responsible for a specific aspect of the backend functionality.

### Features

- GPU Device Management: Manages GPU resources like devices, buffers, and command encoders using WGPU.
- Compute Pass Execution: Implements compute passes for tensor operations, ensuring efficient execution on the GPU.
- Data Linking and Copying: Manages data flow between tensors using GPU buffers and encoders.
- Shader Program Management: Configures and manages shaders for executing tensor computations and operations.
- Data Retrieval: Supports efficient GPU data readout for analysis.

## Modules

- `backend`: Defines the main `Backend` struct that manages the WGPU device and provides methods to manipulate GPU resources.
- `compute`: Implements the `Compute` struct, which is used to manage and execute compute passes on the GPU.
- `limits`: Defines the `Limits` struct, which holds information about the limits of the WGPU device.
- `linker`: Implements the `Linker` struct, which handles copying data between GPU buffers.
- `processor`: Implements the `Processor` struct, which sets up and manages shader programs and their associated resources.
- `readout`: Implements the `Stage` struct, which is responsible for reading out data from the GPU.
- `source`: Defines the `Source` struct, which represents a source of data for the backend.
- `tensor`: Defines the `Tensor` struct, which represents a tensor stored on the GPU and provides methods for tensor operations.
