use pretty_assertions::assert_eq;
use tengu_tensor::Tengu;

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
    let mut add = graph.get_probe::<f32>("main/addmul").unwrap();
    let mut sub = graph.get_probe::<f32>("main/subdiv").unwrap();
    let mut exp = graph.get_probe::<u32>("main/explog").unwrap();

    // Run the computation and display the result twice.
    graph.compute(1).unwrap();
    assert_eq!(add.retrieve().await.unwrap(), [6.0, 13.0, 22.0, 33.0]);
    assert_eq!(sub.retrieve().await.unwrap(), [9.8, 11.0, 12.5, 15.0]);
    assert_eq!(exp.retrieve().await.unwrap(), [4, 9, 22, 56]);
}
