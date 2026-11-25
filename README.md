# Charlotte

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust edition 2024](https://img.shields.io/badge/rust-edition%202024-orange.svg)](Cargo.toml)
[![Rust 1.88+](https://img.shields.io/badge/rust-1.88%2B-orange.svg)](Cargo.toml)

**A web-research agent SDK.** Browser tools, search, and crawl run on a [SuperGlue](https://github.com/pinhole-io/superglue) loop.

Charlotte launches Chromium through [playwright-rs](https://crates.io/crates/playwright-rs). It can also connect to a browser you already run. Optional pieces add [Notepad](https://github.com/pinhole-io/notepad) session memory, Qdrant, and document parsers.

[Quick start](#quick-start) · [Install](#install) · [Search](#search-backend) · [URL policy](#url-policy) · [Examples](#examples)

## Contents

- [Requirements](#requirements)
- [Install](#install)
- [Quick start](#quick-start)
- [Features](#features)
- [Cargo features](#cargo-features)
- [How it fits together](#how-it-fits-together)
- [Browser backends](#browser-backends)
- [Search backend](#search-backend)
- [URL policy](#url-policy)
- [Host integration](#host-integration)
- [Examples](#examples)
- [Library](#library)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Development](#development)
- [License](#license)

## Requirements

- Rust 1.88 or newer. `rust-version` in `Cargo.toml` follows playwright-rs.
- Chromium that matches `playwright_rs::PLAYWRIGHT_VERSION`.
- Network access on the first build so Cargo can fetch [superglue](https://github.com/pinhole-io/superglue) from git.
- A sibling checkout of [jig](https://github.com/pinhole-io/jig). Cargo resolves `path = "../jig"`.
- A sibling checkout of [notepad](https://github.com/pinhole-io/notepad) when you enable `notepad` or the default `full` feature.
- `OPENAI_API_KEY` when you call a live model.

## Install

There is no crates.io package. Clone `charlotte`, `jig`, and `notepad` as siblings. Cargo fetches `superglue` from git.

```bash
mkdir -p ~/src && cd ~/src
git clone https://github.com/pinhole-io/charlotte.git
git clone https://github.com/pinhole-io/jig.git
git clone https://github.com/pinhole-io/notepad.git
```

You need this layout:

```text
~/src/
  charlotte/
  jig/
  notepad/
```

Add the crate from git only after those paths exist on the build machine. The published `Cargo.toml` uses path dependencies.

Install Chromium for the Playwright driver pinned by `playwright-rs` 0.10.0:

```bash
npx playwright@1.58.2 install chromium
```

That version is `playwright_rs::PLAYWRIGHT_VERSION` for this lockfile. After you bump `playwright-rs`, install the version that constant reports.

## Quick start

From this repository:

```bash
export OPENAI_API_KEY=sk-...
cargo run --example charlotte_research -- "Summarize recent Rust release notes. Cite URLs."
```

Interactive research chat:

```bash
cargo run --example charlotte_research -- --interactive
```

Show a headed browser window:

```bash
cargo run --example charlotte_research -- --interactive --headed
```

## Features

| Feature | What you get |
| --- | --- |
| Research agent | `charlotte_agent_spec` is a SuperGlue `AgentSpec` for browser-backed research |
| Browser session | `BrowserHandle` owns one isolated context. Page access is serialized. |
| Browser host | `BrowserHost` owns one browser process and opens isolated sessions |
| Tools | `register_charlotte_tools` adds navigate, search, page read, crawl, and extract |
| Research brief | `expand_user_request_to_research_brief` turns a short request into a brief before tools run |
| Session memory | Notepad tools store large tool output during a run. Requires the `notepad` feature. |
| Ingest | Chunk, validate, and dedupe text. Document parsers require the `documents` feature. |

For a single page after navigate, prefer `page_markdown`. Pass `focus` to rank matching sections. The default length is about 32,000 characters (`CHARLOTTE_MAX_MARKDOWN_CHARS`).

## Cargo features

| Feature | Default | Purpose |
| --- | --- | --- |
| `full` | yes | CLI, document parsers, Notepad, and Qdrant |
| `cli` | through `full` | The `charlotte` binary |
| `documents` | through `full` | PDF, Office, and zip parsers |
| `notepad` | through `full` | Session notepad and offload hooks |
| `qdrant` | through `full` | Qdrant client and tools |
| `synthesize` | no | Notepad synthesis. Also enables `notepad`. |
| `harn` | no | Browser and search surface for an embedding host. Omits the optional components. |

```bash
cargo check
cargo check --no-default-features --features harn
cargo test --features synthesize
```

## How it fits together

```text
Host / CLI / example
        │
        ▼
    Charlotte
        │
        ├── SuperGlue     model loop and tools
        ├── Playwright    Chromium, or an existing CDP browser
        ├── SearXNG       default web search
        └── Notepad       optional session memory
```

- Charlotte owns the browser session, the URL policy, and the research tools.
- SuperGlue owns the model call and the tool loop.
- A host registers tools, then runs `charlotte_agent_spec`.

## Browser backends

| Value | Behavior |
| --- | --- |
| `playwright` | Launch Playwright-managed Chromium. This is the default. |
| `obscura` or `cdp` | Connect to an existing CDP endpoint. |

```bash
export CHARLOTTE_BROWSER_BACKEND=playwright
export CHARLOTTE_CDP_ENDPOINT=ws://127.0.0.1:9222
```

`CHARLOTTE_CDP_ENDPOINT` is required for `obscura` and `cdp`. Point it at a Chromium DevTools endpoint you already run. `ws://` and `http://` forms both work.

## Search backend

`web_search` uses SearXNG JSON (`GET /search?format=json`) by default.

| Variable | Default | Effect |
| --- | --- | --- |
| `CHARLOTTE_SEARCH_BACKEND` | `searxng` | `searxng` or `browser` |
| `CHARLOTTE_SEARXNG_URL` | `http://127.0.0.1:18088` | SearXNG base URL |
| `CHARLOTTE_SEARCH_URL_TEMPLATE` | DuckDuckGo HTML | Used when the backend is `browser` |
| `CHARLOTTE_WEB_SEARCH_COOLDOWN_SECS` | `20` | Minimum gap between completed `web_search` calls |

Set `CHARLOTTE_SEARCH_BACKEND=browser` to scrape HTML search results through the browser. Navigation and page tools always use the browser. Run SearXNG yourself when you keep the default search backend.

## URL policy

**Warning:** `UrlPolicy` is an SSRF guard. Do not disable it on a production path.

- Every policy accepts only HTTP and HTTPS URLs. URLs with user information are rejected.
- `CHARLOTTE_ALLOW_HOST_SUFFIXES` is a comma-separated allow-list of host suffixes.
- `CHARLOTTE_BLOCK_PRIVATE_IPS=1` rejects non-public IP literals and DNS results. `true` is also accepted.
- The browser applies the policy to navigation, redirects, subresources, and isolated contexts.

Build a `UrlPolicy` in code, or call `apply_browser_config_from_env`. Combine this with network policy for a multi-tenant deployment.

Tracker and ad blocking is best-effort. It is not a security boundary.

## Host integration

Build the focused surface:

```bash
cargo check --no-default-features --features harn
```

`charlotte_tool_specs` returns the bounded schemas before a host creates browser resources.

One `BrowserHost` owns the Playwright driver and the browser process. Open no page until the host activates Charlotte. Call `CharlotteToolSet::close` for each runtime. Close `BrowserHost` after every runtime session is closed.

A host such as Harn can start with one approval tool, `enable_headed_browser`. Approval opens one headed context and reveals `web_search` plus `agentic_crawl`. Each activated runtime gets its own cookies, storage, tabs, and crawl state. A child runtime cannot activate Charlotte. Activation stays in memory. A restart or a new conversation starts disabled.

### Credential aliases

**Warning:** `agentic_crawl` can accept a `credential` alias. The model sees the alias. It does not see the secret.

The host implements `ToolRuntimeContext::resolve_credential`. The resolver returns named `CrawlInput` values and the exact HTTPS origins that may receive them. The host checks the requested origin before it reads a secret. Charlotte does not provide a vault.

Charlotte applies these rules:

- The resolver runs inside the crawl timeout and the cancellation boundary.
- `credential` cannot be combined with `input_text` or `inputs`.
- Every navigation and landed page must match an allowed origin.
- Charlotte checks the current origin again before it types a value.
- Charlotte types a credential only into an observed input target.
- Charlotte removes reflected values from page observations and tool results.
- The sidecar actor rejects credential runs.

## Examples

| Example | Needs | What it does |
| --- | --- | --- |
| `charlotte_research` | `OPENAI_API_KEY` and Chromium | Research chat with browser tools |
| `babe_chat` | `OPENAI_API_KEY` and Qdrant | Babe agent and Qdrant tools. No browser. |
| `ingest_raw` | nothing | Chunk raw text. No API key and no browser. |
| `crawl_load.sh` | Chromium, curl, and python3 | Starts `charlotte-crawl` and checks admission, auth, and rejected URLs. |

```bash
cargo run --example ingest_raw
cargo run --example babe_chat
./examples/crawl_load.sh
```

## Library

```rust
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
        std::env::var("OPENAI_API_KEY")?,
        "gpt-5.4-nano-2026-03-17-mini",
    );
    let llm = ToolLlmContext {
        http: http.clone(),
        hooks: hooks.clone(),
        guardrails: guardrails.clone(),
        options: base_opts.clone(),
    };
    register_charlotte_tools(
        &tools,
        &browser,
        &Default::default(),
        &search_config_from_env()?,
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

Keep the API key in the environment. Do not commit it.

## Configuration

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | Model key for live runs |
| `OPENAI_BASE_URL` | Model host. The CLI default is `https://api.openai.com`. |
| `OPENAI_MODEL` | Model id. The CLI default is `gpt-6-luna`. Reasoning effort is `high`. |
| `CHARLOTTE_BRIEF_MODE` | `llm`, `heuristic`, `always`, or `never`. The default is `llm`. |
| `CHARLOTTE_STORAGE_STATE_PATH` | Playwright storage-state JSON |
| `CHARLOTTE_MAX_TEXT_CHARS` | `page_content` cap. The default is 48000. |
| `CHARLOTTE_MAX_MARKDOWN_CHARS` | `page_markdown` cap. The default is 32000. |
| `CHARLOTTE_MAX_HTML_BYTES` | HTML budget before Markdown conversion. The default is 600000. |
| `HTTP_MAX_RETRIES` | Retries after a failed HTTP attempt. The default is 3. |
| `HTTP_RETRY_INITIAL_MS` | First backoff delay. The default is 1000. |
| `HTTP_RETRY_MAX_INTERVAL_MS` | Backoff cap. The default is 6000. |
| `HTTP_RETRY_MULTIPLIER` | Backoff multiplier. The default is 3.0. |
| `HTTP_QUOTA_PER_SECOND` | Optional global request rate. Use an integer of 1 or more. |
| `QDRANT_API_KEY` | Qdrant key for the Qdrant examples |

Browser and search variables are in [Browser backends](#browser-backends) and [Search backend](#search-backend).

## Troubleshooting

| What you see | What to do |
| --- | --- |
| `can't find crate jig` or `notepad` | Clone those repositories next to this one. See [Install](#install). |
| `can't find crate superglue` or git fetch fails | Allow Cargo to reach `https://github.com/pinhole-io/superglue`, then run `cargo fetch`. |
| Missing `OPENAI_API_KEY` | Export the key before a live example. |
| Playwright cannot find a browser | Install Chromium for `playwright_rs::PLAYWRIGHT_VERSION`. |
| Search returns no rows | Start SearXNG at `CHARLOTTE_SEARXNG_URL`, or set `CHARLOTTE_SEARCH_BACKEND=browser`. |
| `CHARLOTTE_CDP_ENDPOINT must be set` | Export the endpoint, or use `CHARLOTTE_BROWSER_BACKEND=playwright`. |
| A page load is rejected | The URL policy blocked it. Check the scheme, the host suffix, and `CHARLOTTE_BLOCK_PRIVATE_IPS`. |
| Ignored tests fail in the default suite | They do not run in `cargo test`. Pass `-- --ignored` when Chromium is installed. |

## Development

```bash
cargo fmt
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --no-default-features --features harn
```

- Unit tests cover chunking, crawl, URL policy, and robots parsing.
- Wiremock tests cover brief expansion, SearXNG mapping, and agentic crawl decisions. They do not need a live key or Chrome.
- Playwright and live CDP tests are `#[ignore]`. Run them with `cargo test -- --ignored` when the browsers are installed.

## License

The license is [MIT](LICENSE).

Keep the copyright notice and the permission notice in every copy.
