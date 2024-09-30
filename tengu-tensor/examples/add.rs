use tengu_tensor::Tengu;

#[tokio::main]
pub async fn main() {
    // Initialize input tensors.
    let tengu = Tengu::new().await.unwrap();
    let a = tengu.tensor([2, 2]).init(&[1.0, 2.0, 3.0, 4.0]);
    let b = tengu.tensor([2, 2]).init(&[5.0, 6.0, 7.0, 8.0]);
    let c = tengu.tensor([2, 2]).init(&[4.0, 3.0, 2.0, 1.0]);

    // Create computation graph.
    let mut graph = tengu.graph();
    graph
        .add_block("main")
        .add_computation("add", a.clone() + b.clone())
        .add_computation("sub", b - a + c);

    // Set up probes.
    let add = graph.probe("main", "add").unwrap();
    let sub = graph.probe("main", "sub").unwrap();

    // Run one step of computation and display the result.
    graph.step();
    println!("{:?}", add.retrieve::<f32>().await.unwrap());
    println!("{:?}", sub.retrieve::<f32>().await.unwrap());
}
