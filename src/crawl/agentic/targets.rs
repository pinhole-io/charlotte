//! Filter and rank observe targets before they go to Jev.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use url::Url;

use crate::browser::url_policy::UrlPolicy;

/// Interactive element returned by the sidecar `observe` / `get_links` command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct RawTarget {
    pub id: String,
    pub kind: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub href: Option<String>,
    /// `nav`, `content`, or `other`. Empty deserializes as `other`.
    #[serde(default)]
    pub region: String,
    /// Nearest preceding heading, clipped to 60 characters.
    #[serde(default)]
    pub section: String,
    #[serde(default)]
    pub in_viewport: bool,
}

/// Links of one page region and heading, in first-seen order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkSection {
    pub key: String,
    pub label: String,
    pub links: Vec<RawTarget>,
}

/// One page snapshot from the sidecar.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct ObservePage {
    #[serde(default)]
    pub title: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub markdown: String,
    #[serde(default)]
    pub targets: Vec<RawTarget>,
    #[serde(default)]
    pub scroll_y: u32,
    #[serde(default)]
    pub viewport_h: u32,
    #[serde(default)]
    pub scroll_height: u32,
    #[serde(default)]
    pub at_top: bool,
    #[serde(default)]
    pub at_bottom: bool,
}

const SCROLL_SLACK: u32 = 8;

impl ObservePage {
    /// True when the viewport is at the bottom (or the page does not scroll).
    #[must_use]
    pub fn at_page_bottom(&self) -> bool {
        if self.at_bottom {
            return true;
        }
        if self.scroll_height == 0 || self.viewport_h == 0 {
            return false;
        }
        self.scroll_y.saturating_add(self.viewport_h) + SCROLL_SLACK >= self.scroll_height
    }

    /// True when the viewport is at the top of the page.
    #[must_use]
    pub fn at_page_top(&self) -> bool {
        self.at_top || self.scroll_y <= SCROLL_SLACK
    }
}

/// Allowed target kinds after filter.
const KINDS: &[&str] = &["link", "button", "input", "select"];

pub const RAW_TARGET_CAP: usize = 1500;
pub const LABEL_CLIP: usize = 80;
pub const SECTION_CLIP: usize = 60;
pub const GOAL_CLIP: usize = 2000;
pub const TITLE_CLIP: usize = 120;
pub const EXCERPT_CLIP: usize = 2000;

/// Truncate to `max` Unicode scalars and add an ellipsis when clipped.
#[must_use]
pub fn clip_chars(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max).collect();
    out.push('…');
    out
}

/// Keep a target when kind, label, and UrlPolicy pass. Origin filter is opt-in.
pub fn filter_targets(
    raw: &[RawTarget],
    policy: &UrlPolicy,
    seed_url: &str,
    same_origin_only: bool,
) -> Vec<RawTarget> {
    let mut out = Vec::new();
    for (index, t) in raw.iter().take(RAW_TARGET_CAP).enumerate() {
        let kind = t.kind.trim().to_ascii_lowercase();
        if !KINDS.contains(&kind.as_str()) {
            continue;
        }
        let label = clip_chars(t.label.trim(), LABEL_CLIP);
        if label.is_empty() {
            continue;
        }
        let href = t
            .href
            .as_deref()
            .map(str::trim)
            .filter(|h| !h.is_empty())
            .map(|h| resolve_href(seed_url, h));
        if let Some(ref url) = href {
            if policy.check(url).is_err() {
                continue;
            }
            if same_origin_only && !same_origin(seed_url, url) {
                continue;
            }
            if is_javascript_href(url) {
                continue;
            }
        }
        let furniture = href.as_deref().is_some_and(is_page_furniture);
        out.push(RawTarget {
            id: if t.id.trim().is_empty() {
                format!("t{}", index + 1)
            } else {
                t.id.trim().to_string()
            },
            kind,
            label,
            href,
            region: if furniture {
                "nav".into()
            } else {
                normalize_region(&t.region)
            },
            // Furniture carries the heading it sat under. Dropping it collapses
            // one edit control per heading into a single nav group.
            section: if furniture {
                String::new()
            } else {
                clip_chars(t.section.trim(), SECTION_CLIP)
            },
            in_viewport: t.in_viewport,
        });
    }
    out
}

/// Score by goal overlap: last path segment counts double, then label.
/// Keep every target. Jev pages the list across requests.
#[must_use]
pub fn rank_targets(goal: &str, filtered: Vec<RawTarget>) -> Vec<RawTarget> {
    let goal_tokens = tokenize_for_rank(goal);
    let mut scored: Vec<(usize, usize, RawTarget)> = filtered
        .into_iter()
        .enumerate()
        .map(|(ord, t)| {
            let label_score = tokenize_for_rank(&t.label)
                .intersection(&goal_tokens)
                .count();
            let path_score = t
                .href
                .as_deref()
                .map(last_path_segment_tokens)
                .map(|seg| seg.intersection(&goal_tokens).count())
                .unwrap_or(0);
            let score = path_score.saturating_mul(2).saturating_add(label_score);
            (score, ord, t)
        })
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    scored.into_iter().map(|(_, _, t)| t).collect()
}

/// Filter then rank. Used for the observed click/type menu only.
#[must_use]
pub fn select_jev_targets(
    raw: &[RawTarget],
    goal: &str,
    policy: &UrlPolicy,
    page_url: &str,
    same_origin_only: bool,
    exclude_hrefs: &HashSet<String>,
) -> Vec<RawTarget> {
    rank_targets(
        goal,
        filter_observed_targets(raw, policy, page_url, same_origin_only, exclude_hrefs),
    )
}

/// Policy-safe observed targets. Buttons without href stay in the click/type list.
#[must_use]
pub fn filter_observed_targets(
    raw: &[RawTarget],
    policy: &UrlPolicy,
    page_url: &str,
    same_origin_only: bool,
    exclude_hrefs: &HashSet<String>,
) -> Vec<RawTarget> {
    filter_targets(raw, policy, page_url, same_origin_only)
        .into_iter()
        .filter(|t| match &t.href {
            Some(href) => {
                !same_document(page_url, href) && !exclude_hrefs.contains(&canonical_page_url(href))
            }
            None => true,
        })
        .collect()
}

/// Distinct followable hrefs. First occurrence wins. No keyword cut.
#[must_use]
pub fn filter_follow_candidates(
    raw: &[RawTarget],
    policy: &UrlPolicy,
    page_url: &str,
    same_origin_only: bool,
    exclude_hrefs: &HashSet<String>,
) -> Vec<RawTarget> {
    let mut seen = HashSet::new();
    filter_targets(raw, policy, page_url, same_origin_only)
        .into_iter()
        .filter(|t| {
            let Some(href) = t.href.as_deref() else {
                return false;
            };
            if same_document(page_url, href) {
                return false;
            }
            let key = canonical_page_url(href);
            if exclude_hrefs.contains(&key) {
                return false;
            }
            seen.insert(key)
        })
        .collect()
}

/// Drop hrefs already followed from this page. They stay available from other pages.
#[must_use]
pub fn without_tried_follows(candidates: Vec<RawTarget>, tried_hrefs: &[String]) -> Vec<RawTarget> {
    let banned: HashSet<String> = tried_hrefs.iter().map(|h| canonical_page_url(h)).collect();
    candidates
        .into_iter()
        .filter(|t| {
            t.href
                .as_deref()
                .is_none_or(|href| !banned.contains(&canonical_page_url(href)))
        })
        .collect()
}

/// Group followable links by `(region, section)`. Content first. Nav and
/// unclassified chrome (`other`) last.
///
/// Grouping is only reading order. Jev does not pick a heading first.
#[must_use]
pub fn group_link_sections(filtered: Vec<RawTarget>) -> Vec<LinkSection> {
    let mut body_order: Vec<(String, String)> = Vec::new();
    let mut nav_order: Vec<(String, String)> = Vec::new();
    let mut buckets: HashMap<(String, String), Vec<RawTarget>> = HashMap::new();
    for t in filtered {
        let region = normalize_region(&t.region);
        let section = t.section.clone();
        let key = (region.clone(), section.clone());
        if !buckets.contains_key(&key) {
            if region == "nav" || region == "other" {
                nav_order.push(key.clone());
            } else {
                body_order.push(key.clone());
            }
        }
        buckets.entry(key).or_default().push(t);
    }
    let mut sections = Vec::new();
    for group_key in body_order.into_iter().chain(nav_order) {
        let Some(links) = buckets.remove(&group_key) else {
            continue;
        };
        let (region, section) = group_key;
        sections.push(LinkSection {
            key: String::new(),
            label: section_label(&region, &section, &links),
            links,
        });
    }
    for (index, section) in sections.iter_mut().enumerate() {
        section.key = format!("s{:02}", index + 1);
    }
    sections
}

/// Flatten grouped links in reading order. Jev pages the list across requests.
#[must_use]
pub fn follow_menu(sections: &[LinkSection]) -> Vec<RawTarget> {
    sections
        .iter()
        .flat_map(|s| s.links.iter().cloned())
        .collect()
}

fn section_label(region: &str, section: &str, links: &[RawTarget]) -> String {
    let samples: Vec<&str> = links
        .iter()
        .map(|t| t.label.as_str())
        .filter(|s| !s.is_empty())
        .take(3)
        .collect();
    let sample = if samples.is_empty() {
        String::new()
    } else {
        format!(" · {}", samples.join(", "))
    };
    if section.is_empty() {
        format!("{region} · {} links{sample}", links.len())
    } else {
        format!("{region} · {section} · {} links{sample}", links.len())
    }
}

/// Page prefixes for media description pages, raw and percent-encoded.
const MEDIA_PAGE_PREFIXES: &[&str] = &[
    "file:", "file%3a", "image:", "image%3a", "media:", "media%3a",
];

/// True when a link operates on the current page instead of leading onward.
///
/// MediaWiki renders page controls as ordinary body links: a section-edit
/// control beside every heading, and a media description page behind every
/// image. A link walk cannot progress through either, so they belong with the
/// navigation chrome. They stay in the menu; they only lose their body slot.
fn is_page_furniture(url: &str) -> bool {
    let Ok(parsed) = Url::parse(url) else {
        return false;
    };
    if parsed
        .query_pairs()
        .any(|(key, _)| key == "action" || key == "veaction")
    {
        return true;
    }
    let Some(page) = parsed.path_segments().and_then(Iterator::last) else {
        return false;
    };
    let page = page.to_ascii_lowercase();
    MEDIA_PAGE_PREFIXES
        .iter()
        .any(|prefix| page.starts_with(prefix))
}

fn normalize_region(raw: &str) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "nav" => "nav".into(),
        "content" => "content".into(),
        _ => "other".into(),
    }
}

/// Host + path + query, no fragment. Used to detect a no-op follow.
#[must_use]
pub fn canonical_page_url(url: &str) -> String {
    let Ok(mut parsed) = Url::parse(url.trim()) else {
        return url.trim().to_string();
    };
    parsed.set_fragment(None);
    let mut out = parsed.to_string();
    if out.ends_with('/') && parsed.path() != "/" {
        out.pop();
    }
    out
}

/// True when both URLs are the same document (fragments ignored).
#[must_use]
pub fn same_document(a: &str, b: &str) -> bool {
    let left = canonical_page_url(a);
    let right = canonical_page_url(b);
    !left.is_empty() && left == right
}

#[must_use]
pub fn same_origin(a: &str, b: &str) -> bool {
    let Ok(ua) = Url::parse(a) else {
        return false;
    };
    let Ok(ub) = Url::parse(b) else {
        return false;
    };
    ua.scheme() == ub.scheme()
        && canonical_host(&ua) == canonical_host(&ub)
        && ua.port_or_known_default() == ub.port_or_known_default()
}

fn canonical_host(url: &Url) -> Option<String> {
    let host = url.host_str()?.trim_end_matches('.').to_ascii_lowercase();
    (!host.is_empty()).then_some(host)
}

fn resolve_href(base: &str, href: &str) -> String {
    if let Ok(abs) = Url::parse(href) {
        return abs.to_string();
    }
    if let Ok(base) = Url::parse(base)
        && let Ok(joined) = base.join(href)
    {
        return joined.to_string();
    }
    href.to_string()
}

fn tokenize(s: &str) -> HashSet<String> {
    s.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() >= 2)
        .map(ToOwned::to_owned)
        .collect()
}

/// Goal tokens for ranking: drop stopwords, add a simple singular form.
fn tokenize_for_rank(s: &str) -> HashSet<String> {
    const STOP: &[&str] = &[
        "page",
        "pages",
        "wiki",
        "wikipedia",
        "the",
        "on",
        "to",
        "from",
        "using",
        "only",
        "in",
        "click",
        "article",
        "link",
        "links",
        "open",
        "try",
        "get",
        "find",
        "reach",
        "for",
        "by",
        "is",
        "as",
        "and",
    ];
    let mut tokens = tokenize(s);
    let extra: Vec<String> = tokens
        .iter()
        .filter(|w| w.len() > 3 && w.ends_with('s') && !w.ends_with("ss"))
        .map(|w| w[..w.len() - 1].to_string())
        .collect();
    tokens.extend(extra);
    tokens.retain(|w| !STOP.contains(&w.as_str()));
    tokens
}

fn is_javascript_href(url: &str) -> bool {
    url.trim().to_ascii_lowercase().starts_with("javascript:")
}

fn stems_eq(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    let (shorter, longer) = if a.len() < b.len() { (a, b) } else { (b, a) };
    longer.len() > 3 && longer == format!("{shorter}s")
}

/// True when every goal token is the last path segment or a simple plural of it.
fn leftover_is_inflection(last: &HashSet<String>, goal: &HashSet<String>) -> bool {
    if last.is_empty() || !last.is_subset(goal) {
        return false;
    }
    goal.iter().all(|g| last.iter().any(|l| stems_eq(l, g)))
}

/// True when the URL path or title matches the crawl goal (token overlap).
///
/// `Dog_food` / “dog food” match. `Sun` does not match “dog food”.
#[must_use]
pub fn page_matches_goal(url: &str, title: &str, goal: &str) -> bool {
    let goal_tokens = tokenize_for_rank(goal);
    if goal_tokens.is_empty() {
        return false;
    }
    let last_seg = last_path_segment_tokens(url);
    if last_seg.len() >= 2 && last_seg.is_subset(&goal_tokens) {
        return true;
    }
    if leftover_is_inflection(&last_seg, &goal_tokens) {
        return true;
    }
    let mut combined = url_path_tokens(url);
    combined.extend(tokenize(title));
    if goal_tokens.is_subset(&combined) {
        return true;
    }
    let overlap = goal_tokens.intersection(&combined).count();
    overlap >= 2 && overlap * 2 >= goal_tokens.len()
}

fn url_path_tokens(url: &str) -> HashSet<String> {
    match Url::parse(url) {
        Ok(parsed) => tokenize(parsed.path()),
        Err(_) => tokenize(url),
    }
}

fn last_path_segment_tokens(url: &str) -> HashSet<String> {
    let path = match Url::parse(url) {
        Ok(parsed) => parsed.path().to_string(),
        Err(_) => url.to_string(),
    };
    let last = path.rsplit('/').find(|s| !s.is_empty()).unwrap_or("");
    tokenize(last)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target(id: &str, kind: &str, label: &str, href: Option<&str>) -> RawTarget {
        RawTarget {
            id: id.into(),
            kind: kind.into(),
            label: label.into(),
            href: href.map(ToOwned::to_owned),
            ..Default::default()
        }
    }

    fn target_in(id: &str, label: &str, href: &str, region: &str, section: &str) -> RawTarget {
        RawTarget {
            id: id.into(),
            kind: "link".into(),
            label: label.into(),
            href: Some(href.into()),
            region: region.into(),
            section: section.into(),
            ..Default::default()
        }
    }

    #[test]
    fn filter_drops_empty_label_and_bad_kind() {
        let policy = UrlPolicy::permissive();
        let raw = vec![
            target("t1", "link", "", Some("https://example.com/a")),
            target("t2", "cookie", "Accept", None),
            target("t3", "button", "Docs", None),
        ];
        let out = filter_targets(&raw, &policy, "https://example.com/", true);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, "t3");
    }

    #[test]
    fn filter_blocks_other_origin_keeps_seen() {
        let policy = UrlPolicy::permissive();
        let raw = vec![
            target("t1", "link", "Other", Some("https://evil.test/x")),
            target("t2", "link", "Seen", Some("https://example.com/seen")),
            target("t3", "link", "New", Some("https://example.com/new")),
        ];
        let out = filter_targets(&raw, &policy, "https://example.com/", true);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id, "t2");
        assert_eq!(out[1].id, "t3");
    }

    #[test]
    fn filter_keeps_in_page_wikipedia_links() {
        let policy = UrlPolicy::permissive();
        let raw = vec![
            target(
                "t1",
                "link",
                "Help",
                Some("https://en.wikipedia.org/wiki/Help:Contents"),
            ),
            target(
                "t2",
                "link",
                "Main page",
                Some("https://en.wikipedia.org/wiki/Main_Page"),
            ),
            target(
                "t3",
                "link",
                "Dog",
                Some("https://en.wikipedia.org/wiki/Dog"),
            ),
            target("t4", "link", "Alert", Some("javascript:void(0)")),
        ];
        let out = filter_targets(&raw, &policy, "https://en.wikipedia.org/wiki/Sun", false);
        let ids: Vec<&str> = out.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(ids, ["t1", "t2", "t3"]);
    }

    #[test]
    fn rank_dogs_beats_nav_chrome() {
        let filtered = vec![
            target(
                "t1",
                "link",
                "Help:Contents",
                Some("https://en.wikipedia.org/wiki/Help:Contents"),
            ),
            target(
                "t2",
                "link",
                "Main page",
                Some("https://en.wikipedia.org/wiki/Main_Page"),
            ),
            target(
                "t3",
                "link",
                "Animal",
                Some("https://en.wikipedia.org/wiki/Animal"),
            ),
            target(
                "t4",
                "link",
                "Dog",
                Some("https://en.wikipedia.org/wiki/Dog"),
            ),
        ];
        let ranked = rank_targets(
            "open the page for the sun in wikipedia, using only in page links try to click to the page on dogs",
            filtered,
        );
        assert_eq!(ranked[0].id, "t4");
    }

    #[test]
    fn rank_finds_dog_food_among_many_links() {
        let mut filtered = Vec::new();
        for i in 1..=80 {
            filtered.push(target(
                &format!("t{i}"),
                "link",
                "Sun",
                Some(&format!("https://en.wikipedia.org/wiki/Sun_{i}")),
            ));
        }
        filtered.push(target(
            "t200",
            "link",
            "Dog food",
            Some("https://en.wikipedia.org/wiki/Dog_food"),
        ));
        let ranked = rank_targets("reach Dog food from Sun", filtered);
        assert_eq!(ranked[0].id, "t200");
    }

    #[test]
    fn rank_prefers_goal_overlap() {
        let filtered = vec![
            target("t1", "link", "Pricing", Some("https://example.com/p")),
            target(
                "t2",
                "link",
                "Rust book docs",
                Some("https://example.com/rust"),
            ),
            target("t3", "button", "Menu", None),
        ];
        let ranked = rank_targets("read the rust book", filtered);
        assert_eq!(ranked[0].id, "t2");
        assert_eq!(ranked.len(), 3);
    }

    #[test]
    fn clip_chars_adds_ellipsis() {
        assert_eq!(clip_chars("abcd", 3), "abc…");
        assert_eq!(clip_chars("ab", 3), "ab");
    }

    #[test]
    fn scroll_bottom_uses_flag_or_metrics() {
        let flagged = ObservePage {
            at_bottom: true,
            ..Default::default()
        };
        assert!(flagged.at_page_bottom());

        let metrics = ObservePage {
            scroll_y: 2000,
            viewport_h: 800,
            scroll_height: 2800,
            ..Default::default()
        };
        assert!(metrics.at_page_bottom());

        let mid = ObservePage {
            scroll_y: 100,
            viewport_h: 800,
            scroll_height: 2800,
            ..Default::default()
        };
        assert!(!mid.at_page_bottom());
    }

    #[test]
    fn scroll_top_uses_flag_or_metrics() {
        let flagged = ObservePage {
            at_top: true,
            scroll_y: 40,
            ..Default::default()
        };
        assert!(flagged.at_page_top());

        let top = ObservePage {
            scroll_y: 4,
            ..Default::default()
        };
        assert!(top.at_page_top());

        let mid = ObservePage {
            scroll_y: 100,
            viewport_h: 800,
            scroll_height: 2800,
            ..Default::default()
        };
        assert!(!mid.at_page_top());
    }

    #[test]
    fn group_link_sections_puts_nav_last() {
        let sections = group_link_sections(vec![
            target_in(
                "t1",
                "Help",
                "https://en.wikipedia.org/wiki/Help:Contents",
                "nav",
                "Navigation menu",
            ),
            target_in(
                "t2",
                "Earth",
                "https://en.wikipedia.org/wiki/Earth",
                "content",
                "Physical characteristics",
            ),
            target_in(
                "t3",
                "Star",
                "https://en.wikipedia.org/wiki/Star",
                "content",
                "Physical characteristics",
            ),
        ]);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].key, "s01");
        assert!(sections[0].label.contains("content"));
        assert!(sections[0].label.contains("Physical characteristics"));
        assert_eq!(sections[0].links.len(), 2);
        assert_eq!(sections[1].key, "s02");
        assert!(sections[1].label.contains("nav"));
    }

    #[test]
    fn group_link_sections_puts_other_chrome_after_content() {
        let sections = group_link_sections(vec![
            target_in(
                "t4",
                "help",
                "https://www.craigslist.org/about/help/safety",
                "other",
                "",
            ),
            target_in(
                "t40",
                "computers",
                "https://www.craigslist.org/search/area/fresno?cat=sya",
                "content",
                "",
            ),
        ]);
        assert_eq!(sections[0].links[0].label, "computers");
        assert_eq!(sections[1].links[0].label, "help");
    }

    #[test]
    fn group_link_sections_keeps_large_heading_whole() {
        let mut raw = Vec::new();
        for i in 1..=45 {
            raw.push(target_in(
                &format!("t{i}"),
                &format!("Link {i}"),
                &format!("https://example.com/p{i}"),
                "content",
                "See also",
            ));
        }
        raw.push(target_in(
            "t99",
            "Earth",
            "https://example.com/earth",
            "content",
            "Physical characteristics",
        ));
        let sections = group_link_sections(raw);
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[0].links.len(), 45);
        assert!(sections[0].label.contains("See also"));
        assert!(sections[0].label.contains("Link 1"));
        assert_eq!(sections[1].key, "s02");
        assert!(sections[1].label.contains("Physical characteristics"));
        let menu = follow_menu(&sections);
        assert_eq!(menu.len(), 46);
        assert_eq!(menu[0].label, "Link 1");
        assert_eq!(menu[44].label, "Link 45");
        assert_eq!(menu[45].label, "Earth");
    }

    #[test]
    fn follow_menu_keeps_late_content_links_in_one_choice() {
        let mut raw = Vec::new();
        for i in 1..=50 {
            raw.push(target_in(
                &format!("t{i}"),
                &format!("Link {i}"),
                &format!("https://example.com/p{i}"),
                "content",
                "",
            ));
        }
        raw.push(target_in(
            "t99",
            "help",
            "https://example.com/help",
            "nav",
            "",
        ));
        let sections = group_link_sections(raw);
        let menu = follow_menu(&sections);
        assert_eq!(menu.len(), 51);
        assert_eq!(menu[49].label, "Link 50");
        assert_eq!(menu[50].label, "help");
    }

    #[test]
    fn follow_menu_keeps_every_remaining_link() {
        let mut raw = Vec::new();
        for i in 1..=300 {
            raw.push(target_in(
                &format!("t{i}"),
                &format!("Link {i}"),
                &format!("https://example.com/p{i}"),
                "content",
                "",
            ));
        }
        let sections = group_link_sections(raw);
        let menu = follow_menu(&sections);
        assert_eq!(menu.len(), 300);
        assert_eq!(menu[0].label, "Link 1");
        assert_eq!(menu[299].label, "Link 300");
    }

    #[test]
    fn without_tried_follows_drops_only_the_opened_href() {
        let candidates = vec![
            target_in(
                "t162",
                "tools",
                "https://www.craigslist.org/search/area/fresno?cat=tla",
                "content",
                "",
            ),
            target_in(
                "t200",
                "computers",
                "https://www.craigslist.org/search/area/fresno?cat=sya",
                "content",
                "",
            ),
        ];
        let tried = vec![canonical_page_url(
            "https://www.craigslist.org/search/area/fresno?cat=tla#search=2~gallery~0",
        )];
        let open = without_tried_follows(candidates, &tried);
        let sections = group_link_sections(open);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].links.len(), 1);
        assert_eq!(sections[0].links[0].label, "computers");
    }

    #[test]
    fn page_controls_and_media_pages_are_furniture() {
        for url in [
            "https://en.wikipedia.org/wiki/File:The_Sun_in_white_light.jpg",
            "https://en.wikipedia.org/wiki/File%3AThe_Sun_in_white_light.jpg",
            "https://en.wikipedia.org/wiki/Image:Old.png",
            "https://en.wikipedia.org/wiki/Help:Editing?action=edit",
            "https://en.wikipedia.org/w/index.php?title=Sun&action=edit&section=20",
            "https://en.wikipedia.org/w/index.php?title=Sun&veaction=edit",
        ] {
            assert!(is_page_furniture(url), "expected furniture: {url}");
        }
        for url in [
            "https://en.wikipedia.org/wiki/Dog",
            "https://en.wikipedia.org/wiki/Help:Contents",
            "https://en.wikipedia.org/wiki/Filesystem",
            "https://en.wikipedia.org/wiki/Timeline_of_life#Mammals",
            "https://en.wikipedia.org/w/index.php?title=Sun&oldid=12345",
        ] {
            assert!(!is_page_furniture(url), "expected a real page: {url}");
        }
    }

    /// The Sun page ranked an infobox image link at the top of its first body
    /// section, so every run burned three steps on it.
    #[test]
    fn furniture_loses_its_body_slot_but_stays_in_the_menu() {
        let policy = UrlPolicy::permissive();
        let raw = vec![
            target_in(
                "t398",
                "The Sun in white light",
                "https://en.wikipedia.org/wiki/File:The_Sun_in_white_light.jpg",
                "content",
                "",
            ),
            target_in(
                "t531",
                "edit",
                "https://en.wikipedia.org/w/index.php?title=Sun&action=edit&section=20",
                "content",
                "Formation",
            ),
            target_in(
                "t721",
                "Photosynthesis",
                "https://en.wikipedia.org/wiki/Photosynthesis",
                "content",
                "Sunlight and life",
            ),
        ];
        let filtered = filter_targets(&raw, &policy, "https://en.wikipedia.org/wiki/Sun", false);
        assert_eq!(filtered[0].region, "nav");
        assert_eq!(filtered[1].region, "nav");
        assert_eq!(filtered[2].region, "content");

        let sections = group_link_sections(filtered);
        assert_eq!(sections[0].links.len(), 1);
        assert_eq!(sections[0].links[0].label, "Photosynthesis");
        assert!(sections[0].label.contains("Sunlight and life"));
        // Both furniture links collapse into one trailing nav group.
        assert_eq!(sections.len(), 2);
        assert_eq!(sections[1].links.len(), 2);
        assert!(sections[1].label.contains("nav"));
    }

    #[test]
    fn filter_preserves_region_and_section() {
        let policy = UrlPolicy::permissive();
        let raw = vec![target_in(
            "t9",
            "Earth",
            "https://en.wikipedia.org/wiki/Earth",
            "content",
            "Physical characteristics",
        )];
        let out = filter_targets(&raw, &policy, "https://en.wikipedia.org/wiki/Sun", false);
        assert_eq!(out[0].region, "content");
        assert_eq!(out[0].section, "Physical characteristics");
    }

    #[test]
    fn goal_matches_dog_food_path_not_sun() {
        assert!(page_matches_goal(
            "https://en.wikipedia.org/wiki/Dog_food",
            "Dog food - Wikipedia",
            "dog food"
        ));
        assert!(page_matches_goal(
            "https://en.wikipedia.org/wiki/Dog_food",
            "Dog food",
            "reach Dog food from Sun"
        ));
        assert!(!page_matches_goal(
            "https://en.wikipedia.org/wiki/Sun",
            "Sun - Wikipedia",
            "dog food"
        ));
        assert!(!page_matches_goal(
            "https://en.wikipedia.org/wiki/Sun",
            "Sun - Wikipedia",
            "reach Dog food from Sun"
        ));
        assert!(page_matches_goal(
            "https://en.wikipedia.org/wiki/Sun",
            "Sun - Wikipedia",
            "Sun"
        ));
        assert!(page_matches_goal(
            "https://en.wikipedia.org/wiki/Dog",
            "Dog - Wikipedia",
            "click to the page on dogs"
        ));
        assert!(!page_matches_goal(
            "https://en.wikipedia.org/wiki/Sun",
            "Sun - Wikipedia",
            "click to the page on dogs"
        ));
        assert!(page_matches_goal(
            "https://www.craigslist.org/view/d/fresno/dell-e7440-laptop",
            "Dell E7440 laptop 4th Gen i5 - computers",
            "find a laptop"
        ));
        assert!(!page_matches_goal(
            "https://www.craigslist.org/area/fresno",
            "craigslist: fresno jobs, apartments, for sale",
            "find a laptop"
        ));
    }

    #[test]
    fn same_document_ignores_fragment() {
        assert!(same_document(
            "https://en.wikipedia.org/wiki/Wikipedia:About",
            "https://en.wikipedia.org/wiki/Wikipedia:About#top"
        ));
        assert!(!same_document(
            "https://en.wikipedia.org/wiki/Wikipedia:About",
            "https://en.wikipedia.org/wiki/Dog"
        ));
    }

    #[test]
    fn same_origin_canonicalizes_case_default_port_and_trailing_dot() {
        assert!(same_origin(
            "https://EXAMPLE.com.:443",
            "https://example.com/login"
        ));
        assert!(!same_origin(
            "https://example.com",
            "https://example.com:8443"
        ));
    }

    #[test]
    fn select_drops_current_page_and_excluded_hrefs() {
        let policy = UrlPolicy::permissive();
        let raw = vec![
            target(
                "t6",
                "link",
                "About Wikipedia",
                Some("https://en.wikipedia.org/wiki/Wikipedia:About#footer"),
            ),
            target(
                "t7",
                "link",
                "Help",
                Some("https://en.wikipedia.org/wiki/Help:Contents"),
            ),
            target(
                "t8",
                "link",
                "Dog",
                Some("https://en.wikipedia.org/wiki/Dog"),
            ),
        ];
        let mut exclude = HashSet::new();
        exclude.insert(canonical_page_url(
            "https://en.wikipedia.org/wiki/Help:Contents",
        ));
        let out = select_jev_targets(
            &raw,
            "page on dogs",
            &policy,
            "https://en.wikipedia.org/wiki/Wikipedia:About",
            false,
            &exclude,
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].id, "t8");
    }

    #[test]
    fn last_hop_dropped_until_a_third_page() {
        let policy = UrlPolicy::permissive();
        let about = "https://en.wikipedia.org/wiki/Wikipedia:About";
        let humans =
            "https://en.wikipedia.org/wiki/Wikipedia:Wikipedia_is_written_by_humans,_for_humans";
        let help = "https://en.wikipedia.org/wiki/Help:Contents";
        let dog = "https://en.wikipedia.org/wiki/Dog";
        let raw = vec![
            target("t6", "link", "About Wikipedia", Some(about)),
            target("t8", "link", "Help", Some(help)),
            target("t9", "link", "Dog", Some(dog)),
        ];
        let mut from_humans = HashSet::new();
        from_humans.insert(canonical_page_url(about));
        let on_humans =
            select_jev_targets(&raw, "page on dogs", &policy, humans, false, &from_humans);
        let ids: Vec<&str> = on_humans.iter().map(|t| t.id.as_str()).collect();
        assert!(!ids.contains(&"t6"));
        assert!(ids.contains(&"t9"));

        let mut from_help = HashSet::new();
        from_help.insert(canonical_page_url(humans));
        let on_help = select_jev_targets(&raw, "page on dogs", &policy, help, false, &from_help);
        let ids: Vec<&str> = on_help.iter().map(|t| t.id.as_str()).collect();
        assert!(ids.contains(&"t6"));
        assert!(ids.contains(&"t9"));
    }

    #[test]
    fn rank_dogs_beats_for_humans_essay() {
        let filtered = vec![
            target(
                "t221",
                "link",
                "Wikipedia is written by humans, for humans",
                Some(
                    "https://en.wikipedia.org/wiki/Wikipedia:Wikipedia_is_written_by_humans,_for_humans",
                ),
            ),
            target(
                "t6",
                "link",
                "About Wikipedia",
                Some("https://en.wikipedia.org/wiki/Wikipedia:About"),
            ),
            target(
                "t90",
                "link",
                "Dog",
                Some("https://en.wikipedia.org/wiki/Dog"),
            ),
        ];
        let ranked = rank_targets(
            "open the page for the sun in wikipedia, using only in page links try to click to the page on dogs",
            filtered,
        );
        assert_eq!(ranked[0].id, "t90");
    }
}
