//! Crawl service process. Each process has its own browser pool and admission slots.

#[tokio::main]
async fn main() {
    init_logging();
    if let Err(error) = charlotte::serve::run().await {
        eprintln!("charlotte-crawl: {error}");
        std::process::exit(1);
    }
}

fn init_logging() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .init();
}
