#!/usr/bin/env bash
# Exercise charlotte-crawl load controls.
#
# The script starts one crawl process with a single admission slot, then checks
# auth, rejected URLs, one successful page, and a burst that receives HTTP 429.
#
# Needs: Playwright Chromium, curl, and python3.
#
#   ./examples/crawl_load.sh
#
# Optional:
#   CHARLOTTE_CRAWL_PORT   listen port (default 11237)
#   CRAWL_LOAD_URL         page used for the live crawl (default https://example.com)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${CHARLOTTE_CRAWL_PORT:-11237}"
TOKEN="${CRAWL_LOAD_TOKEN:-crawl-load-example}"
PAGE_URL="${CRAWL_LOAD_URL:-https://example.com}"
BIND="127.0.0.1:${PORT}"
BASE="http://${BIND}"
BURST="$(mktemp -d)"
PID=""

cleanup() {
    if [[ -n "${PID}" ]]; then
        kill "${PID}" 2>/dev/null || true
        wait "${PID}" 2>/dev/null || true
    fi
    rm -rf "${BURST}"
}
trap cleanup EXIT

if ! command -v curl >/dev/null || ! command -v python3 >/dev/null; then
    echo "crawl_load: curl and python3 are required" >&2
    exit 1
fi

cd "${ROOT}"
cargo build --bin charlotte-crawl --no-default-features --features serve
TARGET="$(cargo metadata --format-version 1 --no-deps | python3 -c 'import json,sys; print(json.load(sys.stdin)["target_directory"])')"
BIN="${TARGET}/debug/charlotte-crawl"

# One slot, so a second overlapping crawl is refused.
# High soft limits keep this example on the admission path.
env \
    CHARLOTTE_CRAWL_BIND="${BIND}" \
    CHARLOTTE_CRAWL_TOKEN="${TOKEN}" \
    CHARLOTTE_ADMISSION_MAX_ACTIVE=1 \
    CHARLOTTE_MAX_PAGES=1 \
    CHARLOTTE_BROWSER_POOL_SIZE=1 \
    CHARLOTTE_PER_DOMAIN_MAX_ACTIVE=1 \
    CHARLOTTE_MEMORY_SOFT_LIMIT_GIB=1024 \
    CHARLOTTE_TASK_SOFT_LIMIT=1000000 \
    "${BIN}" &
PID="$!"

auth=(-H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json")

wait_for_health() {
    local i code
    for i in $(seq 1 90); do
        if ! kill -0 "${PID}" 2>/dev/null; then
            echo "crawl_load: service exited before it was ready" >&2
            exit 1
        fi
        code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "${auth[@]}" "${BASE}/health" 2>/dev/null || true)"
        if [[ "${code}" == "200" ]]; then
            return 0
        fi
        sleep 1
    done
    echo "crawl_load: service did not become ready" >&2
    exit 1
}

expect_status() {
    local name="$1"
    local want="$2"
    shift 2
    local body code
    body="$(mktemp)"
    code="$(curl -sS -o "${body}" -w '%{http_code}' --max-time 30 "$@")"
    if [[ "${code}" != "${want}" ]]; then
        echo "crawl_load: ${name} returned ${code}, expected ${want}" >&2
        cat "${body}" >&2
        rm -f "${body}"
        exit 1
    fi
    echo "ok  ${name} -> ${code}"
    rm -f "${body}"
}

echo "crawl_load: waiting for ${BASE}"
wait_for_health

expect_status "missing token" 401 \
    -H "Content-Type: application/json" \
    -d "{\"urls\":[\"${PAGE_URL}\"]}" \
    "${BASE}/crawl"

expect_status "two urls" 400 \
    "${auth[@]}" \
    -d "{\"urls\":[\"${PAGE_URL}\",\"${PAGE_URL}/other\"]}" \
    "${BASE}/crawl"

expect_status "ftp url" 400 \
    "${auth[@]}" \
    -d '{"urls":["ftp://example.com/file"]}' \
    "${BASE}/crawl"

expect_status "loopback url" 400 \
    "${auth[@]}" \
    -d '{"urls":["http://127.0.0.1/"]}' \
    "${BASE}/crawl"

expect_status "url user info" 400 \
    "${auth[@]}" \
    -d '{"urls":["https://user:secret@example.com/"]}' \
    "${BASE}/crawl"

page_body="$(mktemp)"
page_code="$(curl -sS -o "${page_body}" -w '%{http_code}' --max-time 60 "${auth[@]}" -d "{\"urls\":[\"${PAGE_URL}\"]}" "${BASE}/crawl")"
if [[ "${page_code}" != "200" ]]; then
    echo "crawl_load: page crawl returned ${page_code}" >&2
    cat "${page_body}" >&2
    exit 1
fi
python3 - "${page_body}" <<'PY'
import json
import sys
body = json.load(open(sys.argv[1]))
row = body["results"][0]
if row.get("success") is not True:
    raise SystemExit(f"page crawl was not successful: {row.get('error_message')}")
markdown = row.get("markdown") or ""
if not markdown.strip():
    raise SystemExit("page crawl returned empty markdown")
print(f"ok  page crawl -> 200 ({len(markdown)} markdown chars)")
PY
rm -f "${page_body}"

# Eight crawls at once. Only one slot is free, so at least one request is refused.
echo "crawl_load: burst of 8 (one slot; the rest should return 429)"
burst_pids=()
for i in $(seq 1 8); do
    (
        code="$(curl -sS -D "${BURST}/${i}.headers" -o "${BURST}/${i}.body" -w '%{http_code}' --max-time 60 \
            "${auth[@]}" \
            -d "{\"urls\":[\"${PAGE_URL}\"]}" \
            "${BASE}/crawl" || true)"
        if [[ -z "${code}" ]]; then
            code="curl_failed"
        fi
        printf '%s\n' "${code}" >"${BURST}/${i}.code"
        echo "burst ${i} -> ${code}"
    ) &
    burst_pids+=("$!")
done
# Wait for the burst only. A bare wait also waits for the server.
wait "${burst_pids[@]}"

python3 - "${BURST}" <<'PY'
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
codes = []
retry_after = False
for path in sorted(root.glob("*.code")):
    code = path.read_text().strip()
    codes.append(code)
    if code == "429":
        headers = path.with_suffix(".headers").read_text().lower()
        if "retry-after:" in headers:
            retry_after = True
print("burst statuses:", " ".join(codes))
if "200" not in codes:
    raise SystemExit("burst produced no successful crawl")
if "429" not in codes:
    raise SystemExit("burst did not refuse a crawl while the only slot was busy")
if not retry_after:
    raise SystemExit("429 response did not include Retry-After")
print("ok  burst -> at least one 200 and one 429")
PY

echo "crawl_load: passed"
