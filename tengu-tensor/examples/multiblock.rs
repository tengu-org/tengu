use tengu_tensor::Tengu;

#[tokio::main]
pub async fn main() {
    // Initialize input tensors.
    let tengu = Tengu::new().await.unwrap();
    let a = tengu.tensor([2, 2]).init(&[1.0, 2.0, 3.0, 4.0]);
    let b = tengu.like(&a).label("b").zero();

    // Create computation graph.
    let mut graph = tengu.graph();
    graph.add_block("fst").unwrap().add_computation("out", a + 1.0);
    graph.add_block("snd").unwrap().add_computation("out", b + 1.0);
    graph.link_one("fst/out", "snd/b").unwrap();

    // Set up probes.
    let out = graph.probe("snd/out").unwrap();

    // Run the computation and display the result twice.
    graph
        .process(2, || async {
            println!("{:?}", out.retrieve::<f32>().await.unwrap());
        })
        .await;
}
