use tengu_tensor::{Probable, Tengu};

#[tokio::main]
pub async fn main() {
    let tengu = Tengu::new().await.unwrap();
    let mut graph = tengu.graph();
    let a = tengu.tensor([2, 2]).init(&[1.0, 2.0, 3.0, 4.0]);
    let b = tengu.tensor([2, 2]).init(&[5.0, 6.0, 7.0, 8.0]);
    let result = graph
        .add_block()
        .add_computation(a + b)
        .probe()
        .retrieve()
        .await
        .unwrap();
    println!("{:?}", result);
}
