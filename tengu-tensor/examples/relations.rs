use tengu_tensor::Tengu;

#[tokio::main]
pub async fn main() {
    // Initialize input tensors.
    let tengu = Tengu::new().await.unwrap();
    let a = tengu.tensor([2, 2]).init(&[1.0, 2.0, 3.0, 4.0]);
    let b = tengu.tensor([2, 2]).init(&[1.0, 6.0, 3.0, 8.0]);

    // Create computation graph.
    let mut graph = tengu.graph();
    graph
        .add_block("main")
        .unwrap()
        .add_computation("rel", a.eq(b).cast::<u32>());

    // Set up probes.
    let probe = graph.probe("main/rel").unwrap();

    // Run one step of computation and display the result.
    graph.step();
    println!("{:?}", probe.retrieve::<u32>().await.unwrap());
}
