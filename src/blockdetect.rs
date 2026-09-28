//! Heuristics for bot-block/challenge pages.

/// Return a stable block reason when page content looks like a challenge/block shell.
#[must_use]
pub fn detect_block_page(title: Option<&str>, text: &str) -> Option<&'static str> {
    detect_block_page_with_url(title, text, None)
}

/// Like [`detect_block_page`] but also considers the page URL (e.g. DuckDuckGo HTML search).
#[must_use]
pub fn detect_block_page_with_url(
    title: Option<&str>,
    text: &str,
    page_url: Option<&str>,
) -> Option<&'static str> {
    let title = title.unwrap_or_default().to_ascii_lowercase();
    let text = text.to_ascii_lowercase();
    let trimmed = text.trim();

    let has = |needle: &str| text.contains(needle) || title.contains(needle);
    if superglue::content_quality::has_block_phrase(&text)
        || superglue::content_quality::has_block_phrase(&title)
    {
        if has("hcaptcha") || has("g-recaptcha") || has("cf-turnstile") || has("captcha") {
            return Some("captcha_challenge");
        }
        if has("access denied") || has("request blocked") {
            return Some("access_denied");
        }
        return Some("cloudflare_challenge");
    }

    if has("verification successful. waiting for")
        || has("checking your browser")
        || has("just a moment")
        || has("attention required")
        || has("cf-browser-verification")
    {
        return Some("cloudflare_challenge");
    }
    if has("you've been blocked by network security") {
        return Some("network_security_block");
    }
    if has("access denied") || has("request blocked") {
        return Some("access_denied");
    }
    if has("hcaptcha") || has("g-recaptcha") || has("cf-turnstile") {
        return Some("captcha_challenge");
    }

    let ddg_host = page_url
        .map(str::to_ascii_lowercase)
        .is_some_and(|u| u.contains("duckduckgo.com"));
    if ddg_host
        && (has("unfortunately, bots use duckduckgo")
            || has("bots use duckduckgo too")
            || has("anomaly-modal")
            || has("challenge-form")
            || has("please complete the following challenge")
            || has("select all squares")
            || has("select all images"))
    {
        return Some("duckduckgo_challenge");
    }

    let looks_like_shell_title =
        has("reddit - please") || has("access denied") || has("attention required");
    if looks_like_shell_title && trimmed.chars().count() < 80 {
        return Some("empty_or_js_shell");
    }
    if !trimmed.is_empty() && trimmed.chars().count() < 80 {
        return Some("empty_or_js_shell");
    }
    None
}

#[cfg(test)]
mod tests {
    use super::{detect_block_page, detect_block_page_with_url};

    #[test]
    fn detects_cloudflare_wait_page() {
        assert_eq!(
            detect_block_page(
                Some("Just a moment..."),
                "Verification successful. Waiting for www.allrecipes.com to respond"
            ),
            Some("cloudflare_challenge")
        );
    }

    #[test]
    fn detects_network_security_block() {
        assert_eq!(
            detect_block_page(None, "You've been blocked by network security."),
            Some("network_security_block")
        );
    }

    #[test]
    fn detects_empty_js_shell() {
        assert_eq!(
            detect_block_page(Some("Reddit - Please"), "!"),
            Some("empty_or_js_shell")
        );
    }

    #[test]
    fn detects_duckduckgo_bot_challenge() {
        assert_eq!(
            detect_block_page_with_url(
                Some("DuckDuckGo"),
                "Unfortunately, bots use DuckDuckGo too. Please complete the following challenge.",
                Some("https://html.duckduckgo.com/html/?q=test"),
            ),
            Some("duckduckgo_challenge")
        );
    }

    #[test]
    fn detects_generic_captcha_widgets() {
        assert_eq!(
            detect_block_page(None, "Please verify you are human. hcaptcha widget loaded."),
            Some("captcha_challenge")
        );
    }
}
