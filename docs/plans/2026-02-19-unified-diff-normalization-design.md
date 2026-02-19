# Unified Diff Normalization for Non-OpenAI Models

## Goal

Add a normalization layer to the native engine that detects and strips standard unified diff headers before passing content to the V4A parser, enabling non-OpenAI models to use the `apply_patch` tool.

## Background

The native engine (`engines/native.py`) was designed for OpenAI's Responses API, which pre-parses V4A patches server-side. When non-OpenAI models use this tool, they see the schema `{type, path, diff}` and naturally fill "diff" with standard unified diff format (`--- /dev/null`, `+++ b/path`, `@@ -0,0 +1,N @@`). The V4A parser in `apply_diff.py` (`_parse_create_diff()` line 132) rejects any line not starting with `+`, immediately failing on the `--- /dev/null` header.

For creates: `--- /dev/null` is the first line, not a `+` line, so it raises `ValueError("Invalid Add File Line: --- /dev/null")`.

For updates: `--- a/path` fails context matching, and numeric `@@ -N,M +N,M @@` hunk headers get confused with V4A's `@@ text-anchor` syntax.

## Approach

**Normalization in `native.py` only** (not in `apply_diff.py`). Rationale:

- `apply_diff.py` is a clean port from OpenAI's SDK and should stay a pure V4A parser.
- The native engine already has a pattern for detecting wrong input (V4A marker detection on lines 156-169).
- The normalization is specific to the native engine's use case (models filling in the schema directly).
- Best-effort stripping: remove all recognized unified diff header patterns, pass the rest through. If the V4A parser still rejects it, the existing error messages surface.

## Architecture

**One new function in `native.py`:** `_normalize_unified_diff(diff: str, mode: str) -> str`

**Two call sites updated:** `_create_file()` and `_update_file()` each call the normalizer before passing `diff` to `apply_diff()`.

**No changes to `apply_diff.py`** -- it stays a clean V4A-only parser.

## Components

### `_normalize_unified_diff(diff, mode)`

Processes the diff line-by-line, stripping recognized unified diff headers from the **top of the diff only** (before the first content line). Once a `+`/`-`/` ` content line is seen, stripping stops.

**Patterns to strip:**

| Pattern | Example | Detection |
|---|---|---|
| Git diff header | `diff --git a/foo b/foo` | Line starts with `diff --git ` |
| Old-file header | `--- /dev/null`, `--- a/src/main.py` | Line starts with `--- ` |
| New-file header | `+++ b/src/main.py`, `+++ src/main.py` | Line starts with `+++ ` |
| Numeric hunk header | `@@ -0,0 +1,3 @@`, `@@ -1,5 +1,7 @@ def hello():` | Regex: `^@@ -\d+` |
| No-newline marker | `\ No newline at end of file` | Line starts with `\ ` |

Lines that don't match any pattern pass through untouched to the V4A parser.

### Distinguishing numeric `@@` from V4A text anchors

- Unified diff: `@@ -\d+` (always has minus sign and digits after `@@ `)
- V4A anchor: `@@ ` followed by anything else (source text to match against)
- One regex check: `re.match(r'^@@ -\d+', line)` cleanly separates them.

### The `mode` parameter

In `create` mode, the normalizer is more aggressive -- there should be no `@@` anchors at all in a create diff, so it strips all `@@` lines. In `default` (update) mode, it only strips numeric `@@` lines and preserves V4A text anchors.

### Top-only stripping rule

Headers are only stripped from the top of the diff, before the first `+`/`-`/` ` content line. This avoids false positives if file content happens to contain `--- some text`.

### Call sites: `_create_file()` and `_update_file()`

Each call site passes the diff through `_normalize_unified_diff()` before handing it to `apply_diff()`. The `mode` parameter is set to `"create"` or `"default"` accordingly.

## Data Flow

```
Model sends diff (unified or V4A)
         |
         v
  _create_file() / _update_file()
         |
         v
  _normalize_unified_diff(diff, mode)
    |-- scan lines from top
    |-- strip recognized header patterns
    |-- stop stripping at first content line
    +-- return cleaned diff
         |
         v
  apply_diff()  (existing V4A parser, unchanged)
         |
         v
  File created/updated
```

## Error Handling

**Silent normalization with debug logging.** When `_normalize_unified_diff()` strips lines, it logs at `DEBUG` level (e.g., `"Stripped N unified diff header lines from diff input"`). No user-facing message changes. If the normalized diff still fails in the V4A parser, the existing error messages surface as usual.

No attempt to give "detected unified diff format" errors -- the goal is for it to Just Work.

## Testing Strategy

Tests go in `tests/test_native_engine.py` (existing test file for the native engine).

### Create mode (the user's exact bug)

1. Unified diff with `--- /dev/null` + `+++ b/path` + `@@ -0,0 +1,N @@` headers -- strips headers, content lines pass through, file created correctly.
2. Unified diff with `diff --git` preamble -- strips that too.
3. Already-valid V4A create diff (only `+` lines) -- passes through unchanged (no regression).
4. Mixed: `\ No newline at end of file` at the end -- stripped.

### Update mode

5. Unified diff with `--- a/path` + `+++ b/path` + numeric `@@ -1,5 +1,7 @@` -- strips headers, content lines match via context.
6. V4A update diff with `@@ def hello():` text anchor -- anchor preserved, passes through unchanged (critical no-regression).
7. Numeric `@@ -1,5 +1,7 @@ def hello():` (unified diff with trailing context) -- stripped, but the content/`+`/`-` lines still work.

### Edge cases

8. Empty diff -- passes through unchanged.
9. Diff that's only headers and no content lines -- passes through empty, parser gives its normal error.
10. `+++ ` line inside actual content (after content lines have started) -- NOT stripped due to top-only rule.

## Open Questions

None -- all design decisions have been validated.
