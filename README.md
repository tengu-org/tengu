# Tengu: Real-Time Tensor Graph Simulation Framework

Tengu (short for TENsor GUraph) is a Rust-based framework for building and managing real-time tensor computation graphs, leveraging GPU capabilities through WGPU. Inspired by systems like Simulink, Tengu enables the creation of dynamic, feedback-driven models that simulate complex systems in real-time. It’s suitable for simulating physical mechanisms, ecosystems, market dynamics, multi-agent interactions, and more.
Features

## Features

- Real-Time Execution: Run computation graphs continuously on the GPU using WGPU, enabling real-time feedback and dynamic behavior.
- Flexible Computation Graphs: Create complex models with nodes ("blocks") that support multi-step propagation, making it ideal for simulating time-based and feedback-driven processes.
- Multi-Backend Support: Designed with a modular backend architecture, allowing for different backend implementations such as GPU (via WGPU), CPU, or Vulkan.
- Tensor Manipulation API: Powerful tensor operations including reshaping, slicing, padding, and more, enabling detailed modeling and analysis.
- Asynchronous Data Retrieval: Retrieve data from any node or tensor asynchronously, supporting real-time statistics collection and monitoring without interrupting execution.
- Extensible and Modular Design: Built with extensibility in mind, Tengu allows developers to implement new backends, tensor types, or graph operations easily.

## How Tengu Differs from Conventional Neural Network Frameworks

While Tengu may appear similar to conventional neural network (NN) frameworks due to its tensor and graph-based approach, it fundamentally differs in its design goals and execution model:

1. Continuous Real-Time Execution

    Unlike NN frameworks that typically focus on discrete training and inference steps, Tengu is built for continuous, real-time execution. It runs computation graphs in an unending loop, making it suitable for simulations where feedback and dynamic changes over time are essential, such as multi-agent models, ecosystems, or physical systems.

2. Flexible Computation Nodes (Blocks)

    Tengu uses "blocks" as flexible computation nodes, which can be linked in any manner, including the creation of feedback loops. This flexibility allows for the modeling of non-deterministic behaviors, enabling Tengu to simulate complex systems with dynamic interactions—something that conventional NN frameworks are not optimized for.

3. Feedback-Driven Models

    Neural network frameworks are generally optimized for forward-only computation, without direct support for creating feedback loops within the same execution graph. In contrast, Tengu’s architecture allows nodes to feed outputs back as inputs in subsequent steps, which is crucial for real-world simulations involving causality, feedback control systems, and reactive processes.

4. General-Purpose Simulation Framework

    While conventional NN frameworks focus on optimizing neural network training and inference, Tengu is designed to be a general-purpose simulation framework. It can model a wide range of systems beyond neural networks, including physical processes, economic simulations, and agent-based models. Its goal is to simulate complex behaviors that evolve over time rather than just optimizing models based on pre-defined training data.

5. GPU-Based Execution Without NN Assumptions

    Although Tengu leverages GPU acceleration (via WGPU), it does not assume the constraints of neural networks, such as layer-based architectures or specific activation functions. Instead, it offers a lower-level approach where developers have full control over tensor operations and their interactions within the graph, enabling more customized and varied simulation types.

## Getting Started

### Prerequisites

- Rust: Make sure you have the Rust toolchain installed. Follow instructions here.
- WGPU Support: A compatible GPU and drivers that support WGPU (Vulkan, DX12, Metal, or OpenGL).

### Example Usage

Here's a simple example demonstrating how to create a basic computation graph and execute it:

```rust
use tengu_graph::Tengu;

#[tokio::test]
async fn main() {
    // Initialize input tensors.
    let tengu = Tengu::wgpu().await.unwrap();
    let a = tengu.tensor([2, 2]).label("a").init(&[1.0, 2.0, 3.0, 4.0]);
    let b = tengu.tensor([2, 2]).label("b").init(&[5.0, 6.0, 7.0, 8.0]);
    let c = tengu.tensor([2, 2]).label("c").init(&[5.0, 2.0, 2.0, 4.0]);

    // Create computation graph.
    let mut graph = tengu.graph();
    graph.add_block("fst").unwrap().add_computation("addmul", a.clone() * b.clone() + 1.0)
    graph.add_block("snd").unwrap().add_computation("subdiv", tengu.scalar(2.0) * b.clone() - a.clone() / c);
    graph.add_link("fst/addmul", "snd/subdiv").unwrap(); // fst -> snd
    graph.add_link("snd/subdiv", "fst/a").unwrap();      // snd -> fst

    // Set up probes.
    let add = graph.add_probe::<f32>("main/addmul").unwrap();
    let sub = graph.add_probe::<f32>("main/subdiv").unwrap();

    // Run the computation.
    graph.compute(10).await.unwrap();

    // Retrieve the results and assert.
    let add = add.retrieve().await.unwrap();
    let sub = sub.retrieve().await.unwrap();
}
```

## Use Cases

- Simulating Physical Systems: Build models that simulate physical phenomena like fluid dynamics, mechanical systems, or thermal models.
- Ecosystem and Market Simulations: Model dynamic systems with real-time feedback, such as ecosystems or financial markets.
- Multi-Agent Systems: Develop simulations involving multiple interacting agents, where feedback loops and real-time data are essential.

## Contributing

Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, please feel free to open an issue or a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
