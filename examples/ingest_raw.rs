//! Ingest pipeline demo: **raw text only** — no browser, no LLM, no Qdrant.
//!
//! Runs [`charlotte::ingest`] with [`IngestInput::RawText`](charlotte::ingest::IngestInput) and prints chunk IDs and stats.
//!
//! ```text
//! cargo run --example ingest_raw
//! ```

use charlotte::ingest::{IngestInput, IngestOptions, ingest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let sample = r#"# Demo document

Short paragraph one.

## Section A

More content here with enough text to chunk.

## Section B

Final lines."#;

    let result = ingest(
        None,
        IngestInput::RawText {
            source_id: "demo-doc".to_string(),
            text: sample.to_string(),
        },
        IngestOptions::default(),
    )
    .await?;

    println!("ingest_raw: pages_read={}", result.stats.pages_read);
    println!(
        "chunks: {} produced, {} rejected, {} dupes removed",
        result.stats.chunks_produced, result.stats.chunks_rejected, result.stats.duplicates_removed
    );
    if !result.warnings.is_empty() {
        println!("warnings: {:?}", result.warnings);
    }
    for c in &result.chunks {
        let preview: String = c.text.chars().take(120).collect();
        println!("--- id={} source={} ---\n{preview}...\n", c.id, c.source_id);
    }

    Ok(())
}
