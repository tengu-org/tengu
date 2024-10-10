use tengu_tensor::Tengu;

#[tokio::main]
pub async fn main() {
    // Initialize input tensors.
    let tengu = Tengu::wgpu().await.unwrap();
    let a = tengu.tensor([2, 2]).label("a").init(&[1.0, 2.0, 3.0, 4.0]);
    let b = tengu.tensor([2, 2]).label("b").init(&[5.0, 6.0, 7.0, 8.0]);
    let c = tengu.tensor([2, 2]).label("c").init(&[4.0, 3.0, 2.0, 1.0]);

    // Create computation graph.
    let mut graph = tengu.graph();
    graph
        .add_block("main")
        .unwrap()
        .add_computation("addmul", a.clone() * b.clone() + 1.0)
        .add_computation("subdiv", tengu.scalar(2.0) * b.clone() - a.clone() / c)
        .add_computation("explog", a.exp() + b.log());

    // Set up probes.
    let mut add = graph.get_probe::<f32>("main/addmul").unwrap();
    let mut sub = graph.get_probe::<f32>("main/subdiv").unwrap();
    let mut exp = graph.get_probe::<f32>("main/explog").unwrap();

    // Run the computation and display the result twice.
    graph.compute(1);
    println!("{:?}", add.retrieve().await.unwrap());
    println!("{:?}", sub.retrieve().await.unwrap());
    println!("{:?}", exp.retrieve().await.unwrap());
}
