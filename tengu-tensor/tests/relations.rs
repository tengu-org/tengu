use pretty_assertions::assert_eq;
use tengu_tensor::Tengu;

#[tokio::test]
async fn main() {
    // Initialize input tensors.
    let tengu = Tengu::wgpu().await.unwrap();
    let a = tengu.tensor([2, 2]).init(&[1.0, 2.0, 3.0, 4.0]);
    let b = tengu.tensor([2, 2]).init(&[1.0, 6.0, 3.0, 8.0]);

    // Create computation graph.
    let mut graph = tengu.graph();
    graph
        .add_block("main")
        .unwrap()
        .add_computation("rel", a.eq(b).cast::<u32>());

    // Set up probes.
    let mut probe = graph.get_probe::<u32>("main/rel").unwrap();

    // Run one step of computation and display the result.
    graph.compute(1);
    assert_eq!(probe.retrieve().await.unwrap(), [1, 0, 1, 0]);
}
