# Charlotte

Charlotte is a Rust SDK for a web-research agent.

It builds on [superglue](https://github.com/Bioto/superglue), [notepad](https://github.com/Bioto/notepad), and [playwright-rs](https://crates.io/crates/playwright-rs).

## Hard dependency

Superglue is a required dependency. This crate does not build without it. Notepad is also required, and Charlotte enables notepad's `superglue` feature.

## Sibling checkout layout

`Cargo.toml` depends on sibling crates with `path = "../superglue"` and `path = "../notepad"`.

Clone this repo and those two repos as siblings so the paths resolve:

```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/Bioto/charlotte.git
git clone https://github.com/Bioto/superglue.git
git clone https://github.com/Bioto/notepad.git
```

You need this directory layout:

```text
~/src/
  charlotte/
  superglue/
  notepad/
```

## Install

1. Install a Rust toolchain that meets the crate `rust-version` (**1.88+**, aligned with playwright-rs).
2. Install Chromium that matches the bundled Playwright driver version (`playwright_rs::PLAYWRIGHT_VERSION`):

```bash
npx playwright@<version> install chromium
```

3. Set an LLM API key when you run live examples (for example `OPENAI_API_KEY`).

## Quick start

From the charlotte repo root:

```bash
cargo test
cargo run --example charlotte_research -- "Summarize recent Rust release notes; cite URLs."
```

Interactive research chat:

```bash
cargo run --example charlotte_research -- --interactive
```

Show a headed browser window:

```bash
cargo run --example charlotte_research -- --interactive --headed
```

## What you get

- `charlotte_agent_spec` — a superglue `AgentSpec` for browser-backed research.
- `BrowserHandle` — one shared browser page, mutex-serialized for tool use.
- Tools via `register_charlotte_tools`: navigate, search, page read, crawl, extract, and more.
- Optional `expand_user_request_to_research_brief` — turns a short user message into a research brief before tools run.
- Session notepad tools for large tool-output offload during a run.

For single-page reads after navigate, prefer `page_markdown`. Pass optional `focus` to prioritize matching sections. Default length is about 32k characters (`CHARLOTTE_MAX_MARKDOWN_CHARS`).

## Browser backends

Select the backend at runtime:

| Value | Behavior |
|-------|----------|
| `playwright` | Launch Playwright-managed Chromium (default when `CHARLOTTE_BROWSER_BACKEND` is unset). |
| `obscura` / `cdp` | Connect to an existing CDP endpoint. |

Environment variables:

- `CHARLOTTE_BROWSER_BACKEND=playwright|obscura|cdp`
- `CHARLOTTE_CDP_ENDPOINT` — required for `obscura` / `cdp` (default local CDP endpoint applies when unset)

Point `CHARLOTTE_CDP_ENDPOINT` at a Chromium DevTools endpoint you already run (for example `ws://127.0.0.1:9222` or `http://127.0.0.1:9222`).

## Search backend

`web_search` defaults to **SearXNG** over HTTP JSON (`GET /search?format=json`).

| Variable | Default | Effect |
|----------|---------|--------|
| `CHARLOTTE_SEARCH_BACKEND` | `searxng` | `searxng` or `browser` |
| `CHARLOTTE_SEARXNG_URL` | `http://127.0.0.1:18088` | SearXNG base URL |
| `CHARLOTTE_SEARCH_URL_TEMPLATE` | DuckDuckGo HTML | Used when `CHARLOTTE_SEARCH_BACKEND=browser` |
| `CHARLOTTE_WEB_SEARCH_COOLDOWN_SECS` | `20` | Minimum gap between completed `web_search` calls |

Set `CHARLOTTE_SEARCH_BACKEND=browser` to scrape HTML search results through the browser instead.

Navigation and page tools always use the browser backend. Run a SearXNG instance yourself if you keep the default search backend.

## URL policy (SSRF mitigation)

`UrlPolicy` and `BrowserConfig::url_policy` can restrict which hosts load:

- Set `CHARLOTTE_ALLOW_HOST_SUFFIXES` to a comma-separated allow-list of host suffixes. A non-empty list enforces the allow-list.
- Set `CHARLOTTE_BLOCK_PRIVATE_IPS=1` to reject loopback, private, and link-local IP literals in URLs. Metadata-style hostnames are also blocked when this flag is on.

Build a `UrlPolicy` in code, or call `apply_browser_config_from_env`. Combine this with network policy for multi-tenant deployments. Do not disable these guards for convenience in production paths.

## Cargo examples

| Example | Notes |
|---------|-------|
| `charlotte_research` | Browser tools + optional brief step (`OPENAI_API_KEY`, Chromium). |
| `babe_chat` | Babe agent + Qdrant tools, no browser. |
| `ingest_raw` | Ingest pipeline with raw text only — no API key or browser. |

## Tests

```bash
cargo test
```

- Unit tests cover chunking, crawl, URL policy, robots parsing, and related helpers.
- Wiremock tests cover brief expansion, SearXNG mapping, and agentic crawl decisions without a live API key or Chrome.
- Playwright and live CDP tests are `#[ignore]` by default. Run them with `cargo test -- --ignored` when browsers are installed.

## Library usage

```rust,no_run
use charlotte::{
    BrowserHandle, ToolLlmContext, charlotte_agent_spec, http_client_from_env,
    register_charlotte_tools, search_config_from_env,
};
use superglue::agents::AgentEngine;
use superglue::chat::ChatOptions;
use superglue::guardrails::GuardrailRegistry;
use superglue::hooks::HookRegistry;

async fn demo() -> Result<(), Box<dyn std::error::Error>> {
    let browser = BrowserHandle::from_config(Default::default()).await?;
    let http = http_client_from_env()?;
    let tools = superglue::tools::ToolRegistry::new();
    let hooks = HookRegistry::new();
    let guardrails = GuardrailRegistry::new();
    let base_opts = ChatOptions::new(
        "https://api.openai.com",
        "sk-...",
        "gpt-5.4-nano-2026-03-17-mini",
    );
    let llm = ToolLlmContext {
        http: http.clone(),
        hooks: hooks.clone(),
        guardrails: guardrails.clone(),
        options: base_opts.clone(),
    };
    let search_config = search_config_from_env()?;
    register_charlotte_tools(
        &tools,
        &browser,
        &Default::default(),
        &search_config,
        &http,
        Some(llm),
    )
    .await?;

    let engine = AgentEngine::new(charlotte_agent_spec());
    let _ = engine
        .run(&http, &tools, "Research question…", &base_opts)
        .await?;
    browser.close().await?;
    Ok(())
}
```

## License

This project uses the **BSD-4-Clause** license. See `LICENSE`.

Public materials that mention this software must display the acknowledgement in the LICENSE file.

This license is GPL-incompatible.
