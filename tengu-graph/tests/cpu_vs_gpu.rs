use anyhow::Result;
use pretty_assertions::assert_eq;

use tengu_backend::Backend;
use tengu_graph::{Tengu, CPU, WGPU};

#[tokio::test]
async fn main() {
    let result_wgpu = run::<WGPU>().await.unwrap();
    let result_cpu = run::<CPU>().await.unwrap();
    assert_eq!(result_wgpu, result_cpu);
}

async fn run<B: Backend + 'static>() -> Result<Vec<i32>> {
    // Initialize input tensors.
    let tengu = Tengu::<B>::new().await?;
    let a = tengu.tensor([2, 2]).label("a").init(&[1, 2, 3, 4]);
    let b = tengu.tensor([2, 2]).label("b").init(&[5, 6, 7, 8]);

    // Create computation graph.
    let mut graph = tengu.graph();
    graph.add_block("main").unwrap().add_computation("add", a * b);

    // Set up probes.
    let probe = graph.add_probe::<i32>("main/add")?;

    // Run the computation.
    graph.compute(1).await?;

    // Retrieve the results and assert.
    probe.retrieve().await.map_err(Into::into)
}
