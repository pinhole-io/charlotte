use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;

/// Current UTC timestamp for research prompts (agent spec + brief expansion).
#[must_use]
pub fn utc_now_context_line() -> String {
    let now = OffsetDateTime::now_utc();
    let formatted = now.format(&Rfc3339).unwrap_or_else(|_| now.to_string());
    format!("Current date and time (UTC): {formatted}.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_line_mentions_utc() {
        let line = utc_now_context_line();
        assert!(line.contains("UTC"));
        assert!(line.contains("Current date and time"));
    }
}
