//! HTML → Markdown conversion (MarkItDown-style) with optional focus ranking.

use std::cmp::Ordering;

use htmd::HtmlToMarkdown;
use htmd::options::{HeadingStyle, Options};
use regex::Regex;
use scraper::{Html, Selector};
use serde_json::{Value, json};

use crate::error::CharlotteError;

/// Default markdown length for agent tool calls (~8k tokens). Keeps multi-page sessions within typical model context.
pub const AGENT_DEFAULT_MAX_MARKDOWN_CHARS: usize = 32_000;
/// Upper bound when the model explicitly requests a higher `max_markdown_chars`.
pub const AGENT_MAX_MARKDOWN_CHARS_CAP: usize = 100_000;
/// Default HTML byte budget before Markdown conversion in agent tools.
pub const AGENT_DEFAULT_MAX_HTML_BYTES: usize = 600_000;
/// Upper bound for explicit `max_html_bytes` on agent tools.
pub const AGENT_MAX_HTML_BYTES_CAP: usize = 2_000_000;

const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "this", "to", "was", "what", "when", "where", "which", "who",
    "with",
];

/// Clamp an agent-requested markdown limit to [`AGENT_MAX_MARKDOWN_CHARS_CAP`].
pub fn clamp_agent_markdown_chars(n: usize) -> usize {
    n.clamp(1, AGENT_MAX_MARKDOWN_CHARS_CAP)
}

/// Clamp an agent-requested HTML byte limit to [`AGENT_MAX_HTML_BYTES_CAP`].
pub fn clamp_agent_html_bytes(n: usize) -> usize {
    n.clamp(1, AGENT_MAX_HTML_BYTES_CAP)
}

/// Truncate `s` to at most `max` UTF-8 bytes on a char boundary.
pub fn truncate_str_by_bytes(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Result of MarkItDown-style page conversion.
#[derive(Debug, Clone)]
pub struct PageMarkdownResult {
    pub title: Option<String>,
    pub markdown: String,
    pub truncated: bool,
    pub cleanup: PageMarkdownCleanup,
}

/// Metadata about conversion and optional focus filtering.
#[derive(Debug, Clone)]
pub struct PageMarkdownCleanup {
    pub body_only: bool,
    pub focus_applied: bool,
    pub sections_total: usize,
    pub sections_included: usize,
}

impl PageMarkdownCleanup {
    pub fn to_json(&self) -> Value {
        json!({
            "body_only": self.body_only,
            "focus_applied": self.focus_applied,
            "sections_total": self.sections_total,
            "sections_included": self.sections_included,
        })
    }
}

/// Prepared body HTML and document title from a full HTML document.
#[derive(Debug, Clone)]
pub struct PreparedHtml {
    pub body_html: String,
    pub title: Option<String>,
}

/// Extract `<body>` and `<title>` from a full HTML document; strip script/style (MarkItDown parity).
pub fn prepare_body_html_from_document(html: &str) -> PreparedHtml {
    let doc = Html::parse_document(html);
    let title = document_title(&doc);
    let body_sel = Selector::parse("body").expect("valid body selector");
    let body_html = doc
        .select(&body_sel)
        .next()
        .map(|el| el.html())
        .unwrap_or_else(|| html.to_string());
    PreparedHtml {
        body_html: strip_script_and_style_tags(&body_html),
        title,
    }
}

/// Convert a full HTML **document** to focused Markdown (crawl path).
pub fn document_html_to_markdown(
    html: &str,
    focus: Option<&str>,
    max_html_bytes: usize,
    max_markdown_chars: usize,
) -> Result<PageMarkdownResult, CharlotteError> {
    let truncated_doc = truncate_str_by_bytes(html, max_html_bytes);
    let prepared = prepare_body_html_from_document(truncated_doc);
    body_html_to_markdown(
        &prepared.body_html,
        prepared.title.as_deref(),
        focus,
        max_html_bytes,
        max_markdown_chars,
    )
}

/// Convert a **body** HTML snapshot (`document.body.outerHTML`) to Markdown (browser path).
pub fn body_html_to_markdown(
    body_html: &str,
    title: Option<&str>,
    focus: Option<&str>,
    max_html_bytes: usize,
    max_markdown_chars: usize,
) -> Result<PageMarkdownResult, CharlotteError> {
    let truncated = truncate_str_by_bytes(body_html, max_html_bytes);
    let html_truncated = truncated.len() < body_html.len();
    let cleaned = strip_script_and_style_tags(truncated);
    let md = convert_html_to_markdown(&cleaned)?;
    let (markdown, md_truncated, cleanup) = apply_focus_and_limit(&md, focus, max_markdown_chars);
    Ok(PageMarkdownResult {
        title: title
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string),
        markdown,
        truncated: html_truncated || md_truncated,
        cleanup: PageMarkdownCleanup {
            body_only: true,
            focus_applied: cleanup.focus_applied,
            sections_total: cleanup.sections_total,
            sections_included: cleanup.sections_included,
        },
    })
}

/// Convert HTML to Markdown, skipping script/style/noscript, then truncate to `max_md_chars`.
/// Returns `(markdown, truncated)` where `truncated` is true if output was cut to the limit.
pub fn html_to_markdown_limited(
    html: &str,
    max_md_chars: usize,
) -> Result<(String, bool), CharlotteError> {
    let prepared = prepare_body_html_from_document(html);
    let md = convert_html_to_markdown(&prepared.body_html)?;
    let truncated = md.chars().count() > max_md_chars;
    Ok((truncate_chars_clone(&md, max_md_chars), truncated))
}

/// Split markdown into logical sections (lead + heading-delimited blocks).
pub fn split_markdown_sections(text: &str) -> Vec<String> {
    let normalized = text.replace("\r\n", "\n");
    let lines: Vec<&str> = normalized.split('\n').collect();
    if lines.is_empty() {
        return Vec::new();
    }

    let mut heads: Vec<(usize, usize)> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if let Some(level) = heading_level(line) {
            heads.push((i, level));
        }
    }

    if heads.is_empty() {
        let t = normalized.trim();
        return if t.is_empty() {
            Vec::new()
        } else {
            vec![t.to_string()]
        };
    }

    let mut sections: Vec<String> = Vec::new();
    if heads[0].0 > 0 {
        let pre = lines[..heads[0].0].join("\n");
        if !pre.trim().is_empty() {
            sections.push(pre);
        }
    }
    for i in 0..heads.len() {
        let start = heads[i].0;
        let level = heads[i].1;
        let end_line = heads[(i + 1)..]
            .iter()
            .find(|(_, lev)| *lev <= level)
            .map(|(ln, _)| *ln)
            .unwrap_or(lines.len());
        let body = lines[start..end_line].join("\n");
        if !body.trim().is_empty() {
            sections.push(body);
        }
    }
    sections
}

struct FocusPackResult {
    focus_applied: bool,
    sections_total: usize,
    sections_included: usize,
}

fn apply_focus_and_limit(
    md: &str,
    focus: Option<&str>,
    max_chars: usize,
) -> (String, bool, FocusPackResult) {
    let sections = split_markdown_sections(md);
    let sections_total = sections.len().max(1);

    let focus_terms = focus.map(focus_terms_from_query).filter(|t| !t.is_empty());

    if focus_terms.is_none() {
        let truncated = md.chars().count() > max_chars;
        return (
            truncate_chars_clone(md, max_chars),
            truncated,
            FocusPackResult {
                focus_applied: false,
                sections_total,
                sections_included: sections.len(),
            },
        );
    }

    let terms = focus_terms.unwrap();
    let mut scored: Vec<(usize, f64, usize)> = sections
        .iter()
        .enumerate()
        .map(|(i, sec)| (i, score_section(sec, &terms), section_char_len(sec)))
        .collect();

    // Always include lead section (index 0) when present.
    let mut included: Vec<usize> = Vec::new();
    let mut char_budget = 0usize;
    let mut truncated = false;

    if let Some((0, _, len)) = scored.first().copied() {
        if len <= max_chars {
            included.push(0);
            char_budget += len;
        } else {
            let lead = truncate_chars_clone(&sections[0], max_chars);
            return (
                lead,
                true,
                FocusPackResult {
                    focus_applied: true,
                    sections_total,
                    sections_included: 1,
                },
            );
        }
    }

    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    for (idx, _score, len) in scored {
        if included.contains(&idx) {
            continue;
        }
        let sep = if included.is_empty() { 0 } else { 2 };
        if char_budget + sep + len > max_chars {
            truncated = true;
            continue;
        }
        char_budget += sep + len;
        included.push(idx);
    }

    included.sort_unstable();
    let parts: Vec<String> = included.into_iter().map(|i| sections[i].clone()).collect();
    if parts.is_empty() {
        let truncated = md.chars().count() > max_chars;
        return (
            truncate_chars_clone(md, max_chars),
            truncated,
            FocusPackResult {
                focus_applied: true,
                sections_total,
                sections_included: 0,
            },
        );
    }

    let joined = parts.join("\n\n");
    let out_truncated = truncated || joined.chars().count() > max_chars;
    let sections_included = parts.len();
    (
        truncate_chars_clone(&joined, max_chars),
        out_truncated,
        FocusPackResult {
            focus_applied: true,
            sections_total,
            sections_included,
        },
    )
}

fn convert_html_to_markdown(html: &str) -> Result<String, CharlotteError> {
    let converter = HtmlToMarkdown::builder()
        .skip_tags(vec!["script", "style", "noscript"])
        .options(Options {
            heading_style: HeadingStyle::Atx,
            ..Default::default()
        })
        .build();

    match converter.convert(html) {
        Ok(md) => Ok(post_process_markdown(&md)),
        Err(_) => Ok(post_process_markdown(&html_to_plain_text(html))),
    }
}

fn html_to_plain_text(html: &str) -> String {
    let wrapped = format!("<html><body>{html}</body></html>");
    let doc = Html::parse_document(&wrapped);
    let body_sel = Selector::parse("body").expect("valid body selector");
    doc.select(&body_sel)
        .next()
        .map(|el| el.text().collect::<Vec<_>>().join("\n"))
        .unwrap_or_default()
}

fn document_title(doc: &Html) -> Option<String> {
    let sel = Selector::parse("title").ok()?;
    let t = doc.select(&sel).next()?.text().collect::<String>();
    let t = t.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

fn strip_script_and_style_tags(html: &str) -> String {
    static SCRIPT: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    static STYLE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    let script = SCRIPT
        .get_or_init(|| Regex::new(r"(?is)<script\b[^>]*>.*?</script>").expect("script regex"));
    let style =
        STYLE.get_or_init(|| Regex::new(r"(?is)<style\b[^>]*>.*?</style>").expect("style regex"));
    let s = script.replace_all(html, "");
    style.replace_all(&s, "").to_string()
}

fn post_process_markdown(md: &str) -> String {
    static DATA_IMG: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    static BLANK: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    static BAD_LINK: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();

    let bad = BAD_LINK.get_or_init(|| {
        Regex::new(r"(?i)\[([^\]]*)\]\((?:javascript|data|mailto|tel):[^)]*\)")
            .expect("bad link regex")
    });
    let data_img = DATA_IMG
        .get_or_init(|| Regex::new(r"!\[([^\]]*)\]\(data:[^)]{80,}\)").expect("data img regex"));
    let blank = BLANK.get_or_init(|| Regex::new(r"\n{3,}").expect("blank regex"));

    let mut s = bad.replace_all(md, "$1").to_string();
    s = data_img.replace_all(&s, "![$1](data:...)").to_string();
    s = blank.replace_all(&s, "\n\n").to_string();
    s.trim().to_string()
}

fn heading_level(line: &str) -> Option<usize> {
    let t = line.trim_start();
    if !t.starts_with('#') {
        return None;
    }
    let n = t.chars().take_while(|c| *c == '#').count();
    if !(1..=6).contains(&n) {
        return None;
    }
    match t.chars().nth(n) {
        None => Some(n),
        Some(c) if c.is_whitespace() => Some(n),
        _ => None,
    }
}

fn focus_terms_from_query(focus: &str) -> Vec<String> {
    focus
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .map(str::trim)
        .filter(|w| w.len() >= 3 && !STOPWORDS.contains(w))
        .map(str::to_string)
        .collect()
}

fn score_section(section: &str, terms: &[String]) -> f64 {
    if terms.is_empty() {
        return 0.0;
    }
    let lower = section.to_lowercase();
    let mut score = 0.0;
    let first_line = section.lines().next().unwrap_or("").to_lowercase();
    for term in terms {
        let count = lower.matches(term.as_str()).count();
        if count > 0 {
            score += count as f64;
        }
        if first_line.contains(term.as_str()) {
            score += 2.0;
        }
    }
    score
}

fn section_char_len(s: &str) -> usize {
    s.chars().count()
}

fn truncate_chars_clone(s: &str, max_chars: usize) -> String {
    let n = s.chars().count();
    if n <= max_chars {
        return s.to_string();
    }
    s.chars().take(max_chars).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_to_markdown_heading_and_paragraph() {
        let html = "<html><body><h1>Title</h1><p>Hello world.</p></body></html>";
        let (md, _) = html_to_markdown_limited(html, 10_000).unwrap();
        assert!(md.contains("Title"), "markdown: {md}");
        assert!(md.contains("Hello world"), "markdown: {md}");
    }

    #[test]
    fn strips_script_and_style() {
        let html = "<body><script>x=1</script><style>.a{}</style><p>Keep</p></body>";
        let prepared = prepare_body_html_from_document(&format!("<html>{html}</html>"));
        assert!(!prepared.body_html.contains("<script"));
        assert!(!prepared.body_html.contains("<style"));
        assert!(prepared.body_html.contains("Keep"));
    }

    #[test]
    fn focus_ranks_price_section() {
        let md = "# Product\nIron Flask bottle\n\n## Shipping\nWe ship fast.\n\n## Price\n$29.99 today\n";
        let (focused, _, stats) = apply_focus_and_limit(md, Some("price"), 500);
        assert!(stats.focus_applied);
        assert!(focused.contains("29.99") || focused.contains("Price"));
    }

    #[test]
    fn post_process_strips_javascript_links() {
        let md = "Click [here](javascript:void(0)) now";
        let out = post_process_markdown(md);
        assert!(!out.contains("javascript"));
        assert!(out.contains("here"));
    }

    #[test]
    fn truncate_str_by_bytes_respects_utf8_boundary() {
        let s = "ééé";
        assert_eq!(truncate_str_by_bytes(s, 5), "éé");
    }

    #[test]
    fn clamp_agent_limits() {
        assert_eq!(clamp_agent_markdown_chars(0), 1);
        assert_eq!(clamp_agent_markdown_chars(50_000), 50_000);
        assert_eq!(
            clamp_agent_markdown_chars(500_000),
            AGENT_MAX_MARKDOWN_CHARS_CAP
        );
    }
}
