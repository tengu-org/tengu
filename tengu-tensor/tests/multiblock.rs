use pretty_assertions::assert_eq;
use tengu_tensor::Tengu;

#[tokio::test]
async fn main() {
    // Initialize input tensors.
    let tengu = Tengu::wgpu().await.unwrap();
    let a = tengu.tensor([2, 2]).init(&[1.0, 2.0, 3.0, 4.0]);
    let b = tengu.like(&a).label("b").zero();

    // Create computation graph.
    let mut graph = tengu.graph();
    graph.add_block("fst").unwrap().add_computation("out", a + 1.0);
    graph.add_block("snd").unwrap().add_computation("out", b + 1.0);
    graph.link("fst/out", "snd/b").unwrap();

    // Set up probes.
    let out = graph.get_probe::<f32>("snd/out").unwrap();

    // Run the computation twice.
    graph.compute(2).unwrap();

    // Retrieve results and assert.
    let out: Vec<_> = out.retrieve().await.unwrap().unwrap().into();
    assert_eq!(out, [3.0, 4.0, 5.0, 6.0]);
}
