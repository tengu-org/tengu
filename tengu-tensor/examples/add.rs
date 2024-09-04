use tengu_tensor::Context;

#[tokio::main]
pub async fn main() {
    let context = Context::new().await;
    let a = context.tensor([2, 2]).init(&[1.0, 2.0, 3.0, 4.0]);
    let b = context.tensor([2, 2]).init(&[5.0, 6.0, 7.0, 8.0]);
    let result = (a + b).data().await;
    println!("{:?}", result);
}
