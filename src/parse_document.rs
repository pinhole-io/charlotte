use std::io::Cursor;
use std::path::Path;

use calamine::{Data, Reader, open_workbook_auto_from_rs};
use docx_rs::{
    DocumentChild, InsertChild, ParagraphChild, RunChild, TableCellContent, TableChild,
    TableRowChild,
};
use serde_json::{Value, json};

use crate::error::CharlotteError;

pub const DOCUMENT_MAX_MARKDOWN_CHARS_DEFAULT: usize = 64_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocumentFormat {
    Pdf,
    Html,
    Docx,
    Spreadsheet,
}

impl DocumentFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Pdf => "pdf",
            Self::Html => "html",
            Self::Docx => "docx",
            Self::Spreadsheet => "spreadsheet",
        }
    }
}

fn extension(path: &str) -> Option<String> {
    Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
}

fn zip_contains(bytes: &[u8], name: &str) -> bool {
    let Ok(mut archive) = zip::ZipArchive::new(Cursor::new(bytes)) else {
        return false;
    };
    archive.by_name(name).is_ok()
}

fn detect_format(source: &str, body: &[u8], content_type: Option<&str>) -> Option<DocumentFormat> {
    if body.starts_with(b"%PDF-")
        || content_type.is_some_and(|ct| ct.to_ascii_lowercase().contains("application/pdf"))
    {
        return Some(DocumentFormat::Pdf);
    }
    if body.starts_with(b"\xD0\xCF\x11\xE0") {
        return (extension(source).as_deref() == Some("xls"))
            .then_some(DocumentFormat::Spreadsheet);
    }
    if body.starts_with(b"PK\x03\x04") {
        if zip_contains(body, "word/document.xml") {
            return Some(DocumentFormat::Docx);
        }
        if zip_contains(body, "xl/workbook.xml") {
            return Some(DocumentFormat::Spreadsheet);
        }
    }
    if content_type.is_some_and(|ct| {
        let ct = ct.to_ascii_lowercase();
        ct.contains("text/html") || ct.contains("application/xhtml")
    }) {
        return Some(DocumentFormat::Html);
    }
    match extension(source).as_deref() {
        Some("pdf") => Some(DocumentFormat::Pdf),
        Some("html" | "htm" | "xhtml") => Some(DocumentFormat::Html),
        Some("docx") => Some(DocumentFormat::Docx),
        Some("xlsx" | "xls" | "xlsb" | "ods") => Some(DocumentFormat::Spreadsheet),
        _ => None,
    }
}

fn truncate_markdown(markdown: String, max_markdown_chars: usize) -> (String, bool) {
    let max = max_markdown_chars.max(1);
    let truncated = markdown.chars().count() > max;
    if truncated {
        (markdown.chars().take(max).collect(), true)
    } else {
        (markdown, false)
    }
}

fn spreadsheet_cell(value: &Data) -> String {
    match value {
        Data::Int(value) => value.to_string(),
        Data::Float(value) => value.to_string(),
        Data::String(value) => value.clone(),
        Data::Bool(value) => value.to_string(),
        Data::DateTime(value) => value.to_string(),
        Data::DateTimeIso(value) | Data::DurationIso(value) => value.clone(),
        Data::Error(value) => format!("error: {value:?}"),
        Data::Empty => String::new(),
    }
}

fn spreadsheet_markdown(bytes: &[u8]) -> Result<String, CharlotteError> {
    let mut workbook = open_workbook_auto_from_rs(Cursor::new(bytes))
        .map_err(|e| CharlotteError::msg(format!("spreadsheet parse failed: {e}")))?;
    let mut markdown = String::new();
    for (sheet_name, range) in workbook.worksheets() {
        if !markdown.is_empty() {
            markdown.push_str("\n\n");
        }
        markdown.push_str("## ");
        markdown.push_str(&sheet_name);
        markdown.push('\n');
        for row in range.rows() {
            markdown.push('|');
            for cell in row {
                markdown.push(' ');
                markdown.push_str(&spreadsheet_cell(cell).replace('|', "\\|"));
                markdown.push_str(" |");
            }
            markdown.push('\n');
        }
    }
    Ok(markdown)
}

fn run_text(run: &docx_rs::Run) -> String {
    let mut text = String::new();
    for child in &run.children {
        match child {
            RunChild::Text(value) => text.push_str(&value.text),
            RunChild::Tab(_) | RunChild::PTab(_) => text.push('\t'),
            RunChild::Break(_) | RunChild::CarriageReturn(_) => text.push('\n'),
            RunChild::InstrTextString(value) => text.push_str(value),
            _ => {}
        }
    }
    text
}

fn paragraph_text(paragraph: &docx_rs::Paragraph) -> String {
    let mut text = String::new();
    for child in &paragraph.children {
        match child {
            ParagraphChild::Run(run) => text.push_str(&run_text(run)),
            ParagraphChild::Insert(insert) => {
                for child in &insert.children {
                    if let InsertChild::Run(run) = child {
                        text.push_str(&run_text(run));
                    }
                }
            }
            ParagraphChild::Hyperlink(hyperlink) => {
                for child in &hyperlink.children {
                    if let ParagraphChild::Run(run) = child {
                        text.push_str(&run_text(run));
                    }
                }
            }
            _ => {}
        }
    }
    text
}

fn docx_table_cell(cell: &docx_rs::TableCell) -> String {
    let mut text = String::new();
    for child in &cell.children {
        match child {
            TableCellContent::Paragraph(paragraph) => text.push_str(&paragraph_text(paragraph)),
            TableCellContent::Table(table) => text.push_str(&docx_table(table)),
            _ => {}
        }
    }
    text
}

fn docx_table(table: &docx_rs::Table) -> String {
    let mut markdown = String::new();
    for child in &table.rows {
        let TableChild::TableRow(row) = child;
        markdown.push('|');
        for cell in &row.cells {
            let TableRowChild::TableCell(cell) = cell;
            markdown.push(' ');
            markdown.push_str(&docx_table_cell(cell).replace('|', "\\|"));
            markdown.push_str(" |");
        }
        markdown.push('\n');
    }
    markdown
}

fn docx_markdown(bytes: &[u8]) -> Result<String, CharlotteError> {
    let docx = docx_rs::read_docx(bytes)
        .map_err(|e| CharlotteError::msg(format!("docx parse failed: {e}")))?;
    let mut markdown = String::new();
    for child in &docx.document.children {
        match child {
            DocumentChild::Paragraph(paragraph) => {
                let text = paragraph_text(paragraph);
                if text.trim().is_empty() {
                    markdown.push('\n');
                    continue;
                }
                if let Some(style) = paragraph.property.style.as_ref() {
                    let level = style
                        .val
                        .strip_prefix("Heading")
                        .and_then(|value| value.parse::<usize>().ok())
                        .filter(|level| (1..=6).contains(level));
                    if let Some(level) = level {
                        markdown.push_str(&"#".repeat(level));
                        markdown.push(' ');
                    }
                }
                if paragraph.has_numbering {
                    markdown.push_str("- ");
                }
                markdown.push_str(text.trim_end());
                markdown.push_str("\n\n");
            }
            DocumentChild::Table(table) => {
                markdown.push_str(&docx_table(table));
                markdown.push('\n');
            }
            _ => {}
        }
    }
    Ok(markdown.trim().to_string())
}

/// Parse an in-memory document using its bytes and source filename/URL for type hints.
pub fn parse_document_bytes(
    bytes: &[u8],
    source: &str,
    max_markdown_chars: usize,
) -> Result<Value, CharlotteError> {
    parse_document_bytes_with_content_type(bytes, source, max_markdown_chars, None)
}

fn parse_document_bytes_with_content_type(
    bytes: &[u8],
    source: &str,
    max_markdown_chars: usize,
    content_type: Option<&str>,
) -> Result<Value, CharlotteError> {
    let format = detect_format(source, bytes, content_type);
    let Some(format) = format else {
        return Ok(json!({
            "ok": false,
            "error": "unsupported_document_type",
            "source": source,
            "hint": "Supported document types are PDF, HTML, DOCX, XLSX, XLS, XLSB, and ODS.",
        }));
    };

    let markdown = match format {
        DocumentFormat::Pdf => pdf_extract::extract_text_from_mem(bytes)
            .map_err(|e| CharlotteError::msg(format!("pdf parse failed: {e}")))?,
        DocumentFormat::Html => {
            let html = std::str::from_utf8(bytes)
                .map_err(|e| CharlotteError::msg(format!("HTML is not valid UTF-8: {e}")))?;
            crate::markdown::html_to_markdown_limited(html, max_markdown_chars.max(1))?.0
        }
        DocumentFormat::Docx => docx_markdown(bytes)?,
        DocumentFormat::Spreadsheet => spreadsheet_markdown(bytes)?,
    };
    let (markdown, truncated) = truncate_markdown(markdown, max_markdown_chars);
    let format_name = if format == DocumentFormat::Spreadsheet {
        extension(source).unwrap_or_else(|| format.as_str().to_string())
    } else {
        format.as_str().to_string()
    };
    let mut result = json!({
        "ok": true,
        "source": source,
        "format": format_name,
        "markdown": markdown,
        "truncated": truncated,
        "max_markdown_chars": max_markdown_chars.max(1),
    });
    if format == DocumentFormat::Pdf && result["markdown"].as_str().is_some_and(str::is_empty) {
        result["note"] =
            json!("PDF has no extractable text layer; it may be scanned and require OCR.");
    }
    Ok(result)
}

pub async fn parse_document_url(
    url: &str,
    max_markdown_chars: usize,
) -> Result<Value, CharlotteError> {
    let client = reqwest::Client::new();
    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| CharlotteError::msg(format!("document fetch failed: {e}")))?;
    let status = resp.status().as_u16();
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string);
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| CharlotteError::msg(format!("document read failed: {e}")))?;
    let mut result = parse_document_bytes_with_content_type(
        &bytes,
        url,
        max_markdown_chars,
        content_type.as_deref(),
    )?;
    if let Some(object) = result.as_object_mut() {
        object.insert("url".into(), json!(url));
        object.insert("status".into(), json!(status));
        object.insert("content_type".into(), json!(content_type));
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_pdf_from_header_or_content_type() {
        assert_eq!(
            detect_format("file.bin", b"%PDF-1.7 body", None),
            Some(DocumentFormat::Pdf)
        );
        assert_eq!(
            detect_format("file.bin", b"not pdf", Some("application/pdf")),
            Some(DocumentFormat::Pdf)
        );
        assert_eq!(
            detect_format("file.html", b"<html>", None),
            Some(DocumentFormat::Html)
        );
    }

    #[test]
    fn unsupported_bytes_return_structured_error() {
        let value = parse_document_bytes(b"not a document", "notes.bin", 100).unwrap();
        assert_eq!(value["ok"], false);
        assert_eq!(value["error"], "unsupported_document_type");
    }

    #[test]
    fn parses_html_bytes() {
        let value = parse_document_bytes(
            br#"<html><body><h1>Title</h1><p>Body</p></body></html>"#,
            "page.html",
            1000,
        )
        .unwrap();
        assert_eq!(value["format"], "html");
        assert!(value["markdown"].as_str().unwrap().contains("# Title"));
        assert!(value["markdown"].as_str().unwrap().contains("Body"));
    }
}
