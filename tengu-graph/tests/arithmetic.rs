use pretty_assertions::assert_eq;
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
    graph
        .add_block("main")
        .unwrap()
        .add_computation("addmul", a.clone() * b.clone() + 1.0)
        .add_computation("subdiv", tengu.scalar(2.0) * b.clone() - a.clone() / c)
        .add_computation("explog", (a.exp() + b.log()).cast::<u32>());

    // Set up probes.
    let add = graph.add_probe::<f32>("main/addmul").unwrap();
    let sub = graph.add_probe::<f32>("main/subdiv").unwrap();
    let exp = graph.add_probe::<u32>("main/explog").unwrap();

    // Run the computation.
    graph.compute(1).await.unwrap();

    // Retrieve the results and assert.
    let add = add.retrieve().await.unwrap();
    let sub = sub.retrieve().await.unwrap();
    let exp = exp.retrieve().await.unwrap();
    assert_eq!(add, [6.0, 13.0, 22.0, 33.0]);
    assert_eq!(sub, [9.8, 11.0, 12.5, 15.0]);
    assert_eq!(exp, [4, 9, 22, 56]);
}
