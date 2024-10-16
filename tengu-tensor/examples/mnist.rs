// This examples demonstrates how to create a simple feedforward 3-layer neural network to classify
// handwritten digits from the MNIST dataset.
//
// It includes setting up a second mode for the backpropagation algorithm, training the network,
// and then using the result to predict output.

use tengu_tensor::Tengu;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

#[pollster::main]
pub async fn main() {
    // Initialize tracing.
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env())
        .init();

    // Initialize input tensors.
    let tengu = Tengu::wgpu().await.unwrap();
    let w1 = tengu.tensor([2, 2]).label("w1").init(&[1.0, 2.0, 3.0, 4.0]);

    // Create computation graph.
    let mut graph = tengu.graph();
    graph.add_block("main").unwrap().add_computation("m1", w1);
    //
    // Set up probes.
    let mut p1 = graph.get_probe::<f32>("main/w1").unwrap();

    // Run the computation and display the result twice.
    graph.compute(1).unwrap();
    println!("{:?}", p1.retrieve().await.unwrap());
}
