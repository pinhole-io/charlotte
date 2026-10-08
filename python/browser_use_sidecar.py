# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "browser-use",
# ]
# ///
"""Local Browser Use sidecar for Charlotte agentic_crawl.

Speaks one JSON object per line on stdin/stdout. Does not call System One or Cloud.
Does not read API keys. Connects to an existing CDP browser (Obscura).
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from typing import Any

ALLOWED_PRESS = frozenset({"Enter", "Tab", "Escape"})
MAX_TARGETS = 40
MAX_GET_LINKS = 1500
MAX_MARKDOWN_CHARS = 8000
MAX_WAIT_MS = 5000

OBSERVE_JS = r"""
() => {
  function charlotteRegion(el) {
    if (el.closest('nav, header, footer, aside, [role="navigation"], [role="banner"], [role="contentinfo"], .mw-indicators, .mw-indicator, .hatnote, #toc, .toc, .vector-toc, [role="note"], #leftbar, .leftbar, [id="leftbar"]')) {
      return "nav";
    }
    const box = el.getBoundingClientRect();
    const vw = window.innerWidth || 0;
    const vh = window.innerHeight || 0;
    if (vw > 0 && box.right > 0 && box.right < vw * 0.22) return "nav";
    if (vh > 0 && box.top > vh * 0.88) return "nav";
    if (el.closest('main, [role="main"], article, #center, .col, #table')) {
      return "content";
    }
    return "content";
  }
  function charlotteHeadingText(node) {
    const tag = (node.tagName || "").toUpperCase();
    if (/^H[2-6]$/.test(tag)) {
      return (node.innerText || "").replace(/\s+/g, " ").trim().slice(0, 60);
    }
    return "";
  }
  function charlotteSection(el) {
    let node = el;
    while (node) {
      let prev = node.previousElementSibling;
      while (prev) {
        const own = charlotteHeadingText(prev);
        if (own) return own;
        const heading = prev.querySelector ? prev.querySelector("h2,h3,h4,h5,h6") : null;
        if (heading) {
          const text = charlotteHeadingText(heading);
          if (text) return text;
        }
        prev = prev.previousElementSibling;
      }
      node = node.parentElement;
    }
    return "";
  }
  function charlotteInViewport(r) {
    const vh = window.innerHeight || 0;
    const vw = window.innerWidth || 0;
    return r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw;
  }
  document.querySelectorAll("[data-charlotte-tid]").forEach((el) => {
    el.removeAttribute("data-charlotte-tid");
  });
  const out = [];
  const sel =
    'a[href], button, input, textarea, select, [role="button"], [role="link"]';
  const nodes = document.querySelectorAll(sel);
  let n = 0;
  for (const el of nodes) {
    if (n >= 40) break;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) continue;
    const style = window.getComputedStyle(el);
    if (style.visibility === "hidden" || style.display === "none" || style.opacity === "0") {
      continue;
    }
    const tag = (el.tagName || "").toUpperCase();
    const role = (el.getAttribute("role") || "").toLowerCase();
    let kind = "button";
    if (tag === "A" || role === "link") kind = "link";
    else if (tag === "INPUT" || tag === "TEXTAREA") kind = "input";
    else if (tag === "SELECT") kind = "select";
    else if (tag === "BUTTON" || role === "button") kind = "button";
    const label = (
      el.innerText ||
      el.getAttribute("aria-label") ||
      el.getAttribute("placeholder") ||
      el.getAttribute("value") ||
      el.getAttribute("name") ||
      el.getAttribute("href") ||
      ""
    )
      .replace(/\s+/g, " ")
      .trim();
    if (!label) continue;
    n += 1;
    const id = "t" + n;
    el.setAttribute("data-charlotte-tid", id);
    const href = el.href || el.getAttribute("href") || null;
    out.push({
      id,
      kind,
      label,
      href,
      region: charlotteRegion(el),
      section: charlotteSection(el),
      in_viewport: charlotteInViewport(r),
    });
  }
  const body = document.body ? document.body.innerText || "" : "";
  const y = Math.round(window.scrollY || document.documentElement.scrollTop || 0);
  const vh = Math.round(window.innerHeight || 0);
  const h = Math.round(Math.max(
    document.documentElement.scrollHeight || 0,
    document.body ? document.body.scrollHeight : 0
  ));
  const slack = 8;
  return {
    title: document.title || "",
    url: location.href || "",
    markdown: body.slice(0, 8000),
    targets: out,
    scroll_y: y,
    viewport_h: vh,
    scroll_height: h,
    at_top: y <= slack,
    at_bottom: y + vh >= h - slack,
  };
}
"""

GET_LINKS_JS = r"""
() => {
  function charlotteRegion(el) {
    if (el.closest('nav, header, footer, aside, [role="navigation"], [role="banner"], [role="contentinfo"], .mw-indicators, .mw-indicator, .hatnote, #toc, .toc, .vector-toc, [role="note"], #leftbar, .leftbar, [id="leftbar"]')) {
      return "nav";
    }
    const box = el.getBoundingClientRect();
    const vw = window.innerWidth || 0;
    const vh = window.innerHeight || 0;
    if (vw > 0 && box.right > 0 && box.right < vw * 0.22) return "nav";
    if (vh > 0 && box.top > vh * 0.88) return "nav";
    if (el.closest('main, [role="main"], article, #center, .col, #table')) {
      return "content";
    }
    return "content";
  }
  function charlotteHeadingText(node) {
    const tag = (node.tagName || "").toUpperCase();
    if (/^H[2-6]$/.test(tag)) {
      return (node.innerText || "").replace(/\s+/g, " ").trim().slice(0, 60);
    }
    return "";
  }
  function charlotteSection(el) {
    let node = el;
    while (node) {
      let prev = node.previousElementSibling;
      while (prev) {
        const own = charlotteHeadingText(prev);
        if (own) return own;
        const heading = prev.querySelector ? prev.querySelector("h2,h3,h4,h5,h6") : null;
        if (heading) {
          const text = charlotteHeadingText(heading);
          if (text) return text;
        }
        prev = prev.previousElementSibling;
      }
      node = node.parentElement;
    }
    return "";
  }
  function charlotteInViewport(r) {
    const vh = window.innerHeight || 0;
    const vw = window.innerWidth || 0;
    return r.bottom > 0 && r.top < vh && r.right > 0 && r.left < vw;
  }
  document.querySelectorAll("[data-charlotte-tid]").forEach((el) => {
    el.removeAttribute("data-charlotte-tid");
  });
  const out = [];
  const seen = new Set();
  const nodes = document.querySelectorAll("a[href]");
  let n = 0;
  for (const el of nodes) {
    if (n >= 1500) break;
    const href = el.href || el.getAttribute("href") || "";
    if (!href || href.toLowerCase().startsWith("javascript:")) continue;
    if (seen.has(href)) continue;
    const label = (
      el.innerText ||
      el.getAttribute("aria-label") ||
      el.getAttribute("title") ||
      href ||
      ""
    )
      .replace(/\s+/g, " ")
      .trim();
    if (!label) continue;
    seen.add(href);
    n += 1;
    const id = "t" + n;
    el.setAttribute("data-charlotte-tid", id);
    const r = el.getBoundingClientRect();
    out.push({
      id,
      kind: "link",
      label,
      href,
      region: charlotteRegion(el),
      section: charlotteSection(el),
      in_viewport: charlotteInViewport(r),
    });
  }
  const body = document.body ? document.body.innerText || "" : "";
  const y = Math.round(window.scrollY || document.documentElement.scrollTop || 0);
  const vh = Math.round(window.innerHeight || 0);
  const h = Math.round(Math.max(
    document.documentElement.scrollHeight || 0,
    document.body ? document.body.scrollHeight : 0
  ));
  const slack = 8;
  return {
    title: document.title || "",
    url: location.href || "",
    markdown: body.slice(0, 8000),
    targets: out,
    scroll_y: y,
    viewport_h: vh,
    scroll_height: h,
    at_top: y <= slack,
    at_bottom: y + vh >= h - slack,
  };
}
"""


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def err(req_id: Any, message: str) -> None:
    emit({"ok": False, "id": req_id, "error": message})


async def maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def call_first(obj: Any, names: tuple[str, ...], *args: Any, **kwargs: Any) -> Any:
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return await maybe_await(fn(*args, **kwargs))
    raise RuntimeError(f"none of {names} exist on {type(obj).__name__}")


def decode_evaluate(raw: Any) -> Any:
    """Browser Use `Page.evaluate` always returns a string (JSON for objects)."""
    if raw is None:
        return None
    if isinstance(raw, (dict, list, bool, int, float)):
        return raw
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return raw


async def evaluate(browser: Any, script: str, arg: Any | None = None) -> Any:
    page = await current_page(browser)
    if not hasattr(page, "evaluate"):
        raise RuntimeError("page.evaluate is not available")
    # Must be `(...args) =>` form. Browser Use wraps it as `(fn)(args)`.
    script = script.strip()
    if arg is None:
        raw = await maybe_await(page.evaluate(script))
    else:
        raw = await maybe_await(page.evaluate(script, arg))
    return decode_evaluate(raw)


async def current_page(browser: Any) -> Any:
    for name in ("get_current_page", "current_page", "get_page"):
        fn = getattr(browser, name, None)
        if callable(fn):
            return await maybe_await(fn())
    pages = getattr(browser, "pages", None)
    if pages:
        page_list = await maybe_await(pages) if callable(pages) else pages
        if page_list:
            return page_list[-1]
    raise RuntimeError("could not resolve a Browser Use page")


async def open_owned_tab(browser: Any) -> None:
    for name in ("new_tab", "create_new_tab", "new_page"):
        fn = getattr(browser, name, None)
        if callable(fn):
            try:
                await maybe_await(fn())
                return
            except TypeError:
                await maybe_await(fn("about:blank"))
                return
            except Exception:
                continue


async def connect_browser(cdp_url: str) -> Any:
    from browser_use import Browser

    try:
        browser = Browser(
            cdp_url=cdp_url,
            keep_alive=True,
            is_local=False,
            highlight_elements=False,
        )
    except TypeError:
        browser = Browser(cdp_url=cdp_url)
    start = getattr(browser, "start", None)
    if callable(start):
        await maybe_await(start())
    await open_owned_tab(browser)
    return browser


async def observe_page(browser: Any) -> dict[str, Any]:
    data = await evaluate(browser, OBSERVE_JS.strip())
    if not isinstance(data, dict):
        kind = type(data).__name__
        preview = repr(data)[:120]
        raise RuntimeError(f"observe script returned a non-object ({kind}): {preview}")
    targets = data.get("targets") or []
    if len(targets) > MAX_TARGETS:
        targets = targets[:MAX_TARGETS]
    markdown = str(data.get("markdown") or "")
    if len(markdown) > MAX_MARKDOWN_CHARS:
        markdown = markdown[:MAX_MARKDOWN_CHARS]
    return {
        "title": str(data.get("title") or ""),
        "url": str(data.get("url") or ""),
        "markdown": markdown,
        "targets": targets,
        "scroll_y": int(data.get("scroll_y") or 0),
        "viewport_h": int(data.get("viewport_h") or 0),
        "scroll_height": int(data.get("scroll_height") or 0),
        "at_top": bool(data.get("at_top")),
        "at_bottom": bool(data.get("at_bottom")),
    }


async def get_links_page(browser: Any) -> dict[str, Any]:
    data = await evaluate(browser, GET_LINKS_JS.strip())
    if not isinstance(data, dict):
        kind = type(data).__name__
        preview = repr(data)[:120]
        raise RuntimeError(f"get_links script returned a non-object ({kind}): {preview}")
    targets = data.get("targets") or []
    if len(targets) > MAX_GET_LINKS:
        targets = targets[:MAX_GET_LINKS]
    markdown = str(data.get("markdown") or "")
    if len(markdown) > MAX_MARKDOWN_CHARS:
        markdown = markdown[:MAX_MARKDOWN_CHARS]
    return {
        "title": str(data.get("title") or ""),
        "url": str(data.get("url") or ""),
        "markdown": markdown,
        "targets": targets,
        "scroll_y": int(data.get("scroll_y") or 0),
        "viewport_h": int(data.get("viewport_h") or 0),
        "scroll_height": int(data.get("scroll_height") or 0),
        "at_top": bool(data.get("at_top")),
        "at_bottom": bool(data.get("at_bottom")),
    }


async def click_tid(browser: Any, target_id: str) -> None:
    js = """
    (id) => {
      const el = document.querySelector('[data-charlotte-tid="' + id + '"]');
      if (!el) return false;
      el.click();
      return true;
    }
    """
    ok = await evaluate(browser, js, target_id)
    if not ok:
        raise RuntimeError(f"unknown target_id: {target_id}")


async def type_tid(browser: Any, target_id: str, text: str) -> None:
    js = """
    ([id, text]) => {
      const el = document.querySelector('[data-charlotte-tid="' + id + '"]');
      if (!el) return false;
      el.focus();
      if ("value" in el) el.value = "";
      el.dispatchEvent(new Event("input", { bubbles: true }));
      if ("value" in el) {
        el.value = text;
      } else {
        el.textContent = text;
      }
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
      return true;
    }
    """
    ok = await evaluate(browser, js, [target_id, text])
    if not ok:
        raise RuntimeError(f"unknown target_id: {target_id}")


async def press_key(browser: Any, key: str) -> None:
    page = await current_page(browser)
    keyboard = getattr(page, "keyboard", None)
    if keyboard is not None and hasattr(keyboard, "press"):
        await maybe_await(keyboard.press(key))
        return
    js = """
    (key) => {
      document.activeElement &&
        document.activeElement.dispatchEvent(
          new KeyboardEvent("keydown", { key, bubbles: true })
        );
      return true;
    }
    """
    await evaluate(browser, js, key)


async def scroll_page(browser: Any, delta: int) -> None:
    js = """
    (dy) => {
      window.scrollBy(0, dy);
      return true;
    }
    """
    await evaluate(browser, js, delta)


async def navigate_to(browser: Any, url: str) -> None:
    try:
        await call_first(browser, ("navigate_to", "goto", "navigate"), url)
        return
    except Exception:
        pass
    page = await current_page(browser)
    await call_first(page, ("goto", "navigate", "navigate_to"), url)


async def close_owned_tab(browser: Any) -> None:
    page = await current_page(browser)
    for name in ("close",):
        fn = getattr(page, name, None)
        if callable(fn):
            try:
                await maybe_await(fn())
            except Exception:
                return
            return


async def handle(browser: Any | None, req: dict[str, Any]) -> Any:
    cmd = req.get("cmd")
    req_id = req.get("id")
    if cmd == "start":
        cdp_url = req.get("cdp_url") or "http://127.0.0.1:9222"
        browser = await connect_browser(str(cdp_url))
        emit({"ok": True, "id": req_id})
        return browser
    if browser is None:
        raise RuntimeError("call start first")
    if cmd == "navigate":
        url = req.get("url")
        if not url:
            raise RuntimeError("missing url")
        await navigate_to(browser, str(url))
        page = await observe_page(browser)
        emit({"ok": True, "id": req_id, **page})
        return browser
    if cmd == "observe":
        page = await observe_page(browser)
        emit({"ok": True, "id": req_id, **page})
        return browser
    if cmd == "click":
        tid = req.get("target_id")
        if not tid:
            raise RuntimeError("missing target_id")
        await click_tid(browser, str(tid))
        page = await observe_page(browser)
        emit({"ok": True, "id": req_id, **page})
        return browser
    if cmd == "type":
        tid = req.get("target_id")
        text = req.get("text")
        if not tid:
            raise RuntimeError("missing target_id")
        if text is None:
            raise RuntimeError("missing text")
        await type_tid(browser, str(tid), str(text))
        page = await observe_page(browser)
        emit({"ok": True, "id": req_id, **page})
        return browser
    if cmd == "press":
        key = str(req.get("key") or "")
        if key not in ALLOWED_PRESS:
            raise RuntimeError(f"key not allowed: {key}")
        await press_key(browser, key)
        page = await observe_page(browser)
        emit({"ok": True, "id": req_id, **page})
        return browser
    if cmd == "wait":
        ms = int(req.get("ms") or 0)
        ms = max(0, min(ms, MAX_WAIT_MS))
        await asyncio.sleep(ms / 1000)
        page = await observe_page(browser)
        emit({"ok": True, "id": req_id, **page})
        return browser
    if cmd == "follow":
        tid = req.get("target_id")
        if not tid:
            raise RuntimeError("missing target_id")
        await click_tid(browser, str(tid))
        page = await observe_page(browser)
        emit({"ok": True, "id": req_id, **page})
        return browser
    if cmd == "get_links":
        page = await get_links_page(browser)
        emit({"ok": True, "id": req_id, **page})
        return browser
    if cmd == "scroll":
        delta = int(req.get("delta") or 0)
        await scroll_page(browser, delta)
        page = await observe_page(browser)
        emit({"ok": True, "id": req_id, **page})
        return browser
    if cmd == "stop":
        await close_owned_tab(browser)
        emit({"ok": True, "id": req_id})
        return None
    raise RuntimeError(f"unknown cmd: {cmd}")


async def main() -> None:
    browser: Any | None = None
    try:
        for line in sys.stdin:
            raw = line.strip()
            if not raw:
                continue
            try:
                req = json.loads(raw)
            except json.JSONDecodeError as exc:
                err(None, f"invalid json: {exc}")
                continue
            req_id = req.get("id") if isinstance(req, dict) else None
            try:
                if not isinstance(req, dict):
                    raise RuntimeError("request must be an object")
                browser = await handle(browser, req)
            except Exception as exc:
                err(req_id, str(exc))
    finally:
        if browser is not None:
            stop = getattr(browser, "stop", None)
            if callable(stop):
                try:
                    await maybe_await(stop())
                except Exception:
                    pass


if __name__ == "__main__":
    asyncio.run(main())
