use tengu_tensor::Tengu;

#[tokio::main]
pub async fn main() {
    // Initialize input tensors.
    let tengu = Tengu::new().await.unwrap();
    let a = tengu.tensor([2, 2]).init(&[1.0, 2.0, 3.0, 4.0]);
    let b = tengu.tensor([2, 2]).init(&[5.0, 6.0, 7.0, 8.0]);

    // Create and run computation graph.
    let mut graph = tengu.graph();
    graph.add_block("addition").add_computation("out", a + b);
    graph.compute();

    // Probe the result.
    let result = graph.probe("addition", "out").unwrap();
    println!("{:?}", result.retrieve::<f32>().await.unwrap());
}
