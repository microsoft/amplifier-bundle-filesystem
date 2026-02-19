"""Native engine for the apply_patch tool.

Receives pre-parsed operations from the OpenAI provider. Each call has
{type, path, diff} — the API already decomposed the patch into individual
operations, so no V4A parsing is needed here.

Operation types:
  - create_file: diff contains all +lines -> apply_diff in create mode -> write
  - update_file: read existing -> apply_diff(existing, diff) -> write
  - delete_file: remove file
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from amplifier_core import ModuleCoordinator, ToolResult

from ..apply_diff import apply_diff

logger = logging.getLogger(__name__)

# V4A wrapper markers that should NOT appear in the native engine's diff input.
# If present, the model sent V4A-wrapped content instead of raw hunks.
_V4A_MARKERS = (
    "*** Begin Patch",
    "*** Update File:",
    "*** Add File:",
    "*** Delete File:",
)

# Maximum lines to include in self-healing error responses.
# Keeps error messages practical for models without overwhelming context.
_MAX_CONTENT_HINT_LINES = 200


def _format_content_hint(content: str, path: str) -> str:
    """Format file content for error messages so models can self-correct.

    When a patch fails (context mismatch, file already exists), including the
    current file content lets the model construct a correct diff on the next
    attempt — breaking retry loops without a separate read_file round-trip.
    """
    if not content:
        return "\n\nFile is empty (0 lines)."

    lines = content.splitlines()
    total = len(lines)

    if total <= _MAX_CONTENT_HINT_LINES:
        numbered = "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(lines))
    else:
        head_n = _MAX_CONTENT_HINT_LINES * 2 // 3
        tail_n = _MAX_CONTENT_HINT_LINES - head_n
        head = "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(lines[:head_n]))
        tail_start = total - tail_n
        tail = "\n".join(
            f"{i + 1}\t{line}" for i, line in enumerate(lines[tail_start:], tail_start)
        )
        omitted = total - head_n - tail_n
        numbered = f"{head}\n... [{omitted} lines omitted] ...\n{tail}"

    return f"\n\nCurrent content of {path} ({total} lines):\n{numbered}"


def _normalize_unified_diff(diff: str, mode: str) -> str:
    """Strip standard unified diff headers so the V4A parser can handle the content.

    Processes line-by-line. Header patterns (diff --git, ---, +++, numeric @@)
    are stripped only from the top of the diff (before the first +/-/space content
    line). The ``\\ `` no-newline marker is stripped everywhere.

    Args:
        diff: Raw diff string (may be unified or V4A format).
        mode: "create" strips ALL @@ lines. "default" only strips numeric @@ lines,
              preserving V4A text anchors like ``@@ def hello():``.
    """
    lines = diff.split("\n")
    result: list[str] = []
    in_content = False
    stripped = 0

    for line in lines:
        # Always strip "\ No newline at end of file" markers (can appear anywhere)
        if line.startswith("\\ "):
            stripped += 1
            continue

        # Once we've seen a content line, pass everything through unchanged
        if in_content:
            result.append(line)
            continue

        # --- Top-only header stripping (before first content line) ---

        # Git diff preamble: "diff --git a/foo b/foo"
        if line.startswith("diff --git "):
            stripped += 1
            continue

        # Old-file header: "--- /dev/null" or "--- a/path"
        if line.startswith("--- "):
            stripped += 1
            continue

        # New-file header: "+++ b/path"
        if line.startswith("+++ "):
            stripped += 1
            continue

        # Numeric hunk header: "@@ -0,0 +1,3 @@" or "@@ -1,5 +1,7 @@ def hello():"
        if re.match(r"^@@ -\d+", line):
            stripped += 1
            continue

        # In create mode, strip ALL @@ lines (no V4A anchors expected in creates)
        if mode == "create" and line.startswith("@@"):
            stripped += 1
            continue

        # Content line: +, -, or space prefix marks start of actual diff content
        if line and line[0] in ("+", "-", " "):
            in_content = True
            result.append(line)
            continue

        # Unrecognized line (e.g., V4A text anchor like "@@ def hello():"): pass through
        result.append(line)

    if stripped:
        logger.debug("Stripped %d unified diff header lines from diff input", stripped)

    return "\n".join(result)


class NativeEngine:
    """Native engine: handles pre-parsed operations from the Responses API.

    Schema: {type: "create_file"|"update_file"|"delete_file", path: str, diff?: str}
    Output: git-style status line ("M src/main.py")
    """

    def __init__(self, config: dict[str, Any], coordinator: ModuleCoordinator) -> None:
        self.config = config
        self.coordinator = coordinator
        self.working_dir = config.get("working_dir")
        default_allowed = [self.working_dir] if self.working_dir else ["."]
        self.allowed_write_paths = config.get("allowed_write_paths", default_allowed)
        self.denied_write_paths = config.get("denied_write_paths", [])

    @property
    def description(self) -> str:
        return (
            "Apply pre-parsed file operations from the Responses API. "
            "Each call handles one operation: create_file, update_file, or delete_file."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["create_file", "update_file", "delete_file"],
                    "description": "The operation type",
                },
                "path": {
                    "type": "string",
                    "description": "Relative file path",
                },
                "diff": {
                    "type": "string",
                    "description": "The diff content (not needed for delete_file)",
                },
            },
            "required": ["type", "path"],
        }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute a single pre-parsed operation."""
        op_type = input.get("type")
        path = input.get("path")
        diff = input.get("diff", "")

        if not op_type:
            return ToolResult(
                success=False,
                error={
                    "message": "type is required (create_file, update_file, or delete_file)"
                },
            )
        if not path:
            return ToolResult(
                success=False,
                error={"message": "path is required"},
            )

        resolved = self._resolve_path(path)
        allowed, error_msg = self._validate_path(resolved)
        if not allowed:
            return ToolResult(
                success=False,
                error={"message": error_msg or "Access denied"},
            )

        if op_type == "create_file":
            return await self._create_file(resolved, path, diff)
        elif op_type == "update_file":
            return await self._update_file(resolved, path, diff)
        elif op_type == "delete_file":
            return await self._delete_file(resolved, path)
        else:
            return ToolResult(
                success=False,
                error={"message": f"Unknown operation type: {op_type}"},
            )

    async def _create_file(
        self, resolved: Path, rel_path: str, diff: str
    ) -> ToolResult:
        """Create a new file. Diff contains +lines."""
        # Detect V4A wrapper markers before attempting to apply the diff.
        for marker in _V4A_MARKERS:
            if marker in diff:
                return ToolResult(
                    success=False,
                    error={
                        "message": (
                            f"The diff parameter contains V4A wrapper markers ('{marker}') "
                            "which should not be included. The native engine expects only "
                            "raw diff hunks (@@, +/-, context lines). "
                            "The V4A envelope is handled automatically — "
                            "provide only the diff content for this file."
                        )
                    },
                )

        # Reject create_file if file already exists — the model should use update_file instead.
        # Without this check, the model gets a success signal and may loop indefinitely.
        # Include current content so the model can immediately craft an update_file diff.
        if resolved.exists():
            try:
                current = resolved.read_text(encoding="utf-8")
                content_hint = _format_content_hint(current, rel_path)
            except OSError:
                content_hint = ""
            return ToolResult(
                success=False,
                error={
                    "message": (
                        f"File already exists: {rel_path}. "
                        "Use update_file to modify existing files."
                        f"{content_hint}"
                    )
                },
            )

        diff = _normalize_unified_diff(diff, "create")

        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            content = apply_diff("", diff, mode="create")
            resolved.write_text(content, encoding="utf-8")
            await self._emit_event("apply-patch:applied", rel_path, "create")
            return ToolResult(success=True, output=f"A {rel_path}")
        except ValueError as e:
            return ToolResult(
                success=False, error={"message": f"Create failed for {rel_path}: {e}"}
            )
        except OSError as e:
            return ToolResult(
                success=False, error={"message": f"OS error creating {rel_path}: {e}"}
            )

    async def _update_file(
        self, resolved: Path, rel_path: str, diff: str
    ) -> ToolResult:
        """Update an existing file with a diff."""
        if not resolved.exists():
            return ToolResult(
                success=False,
                error={"message": f"File not found: {rel_path}"},
            )

        # Detect V4A wrapper markers before attempting to apply the diff.
        # Models sometimes send the full V4A envelope (*** Begin Patch, etc.)
        # when they should send only raw diff hunks (@@, +/-, context lines).
        for marker in _V4A_MARKERS:
            if marker in diff:
                return ToolResult(
                    success=False,
                    error={
                        "message": (
                            f"The diff parameter contains V4A wrapper markers ('{marker}') "
                            "which should not be included. The native engine expects only "
                            "raw diff hunks (@@, +/-, context lines). "
                            "The V4A envelope is handled automatically — "
                            "provide only the diff content for this file."
                        )
                    },
                )

        # Read file before the try so `existing` is always in scope for error handling.
        try:
            existing = resolved.read_text(encoding="utf-8")
        except OSError as e:
            return ToolResult(
                success=False,
                error={"message": f"OS error reading {rel_path}: {e}"},
            )

        diff = _normalize_unified_diff(diff, "default")

        try:
            updated = apply_diff(existing, diff)
            resolved.write_text(updated, encoding="utf-8")
            await self._emit_event("apply-patch:applied", rel_path, "update")
            return ToolResult(success=True, output=f"M {rel_path}")
        except ValueError as e:
            # Include current content so the model can construct a correct diff.
            content_hint = _format_content_hint(existing, rel_path)
            return ToolResult(
                success=False,
                error={
                    "message": (
                        f"Context mismatch in {rel_path}: {e} — "
                        "your diff does not match the current file."
                        f"{content_hint}\n\n"
                        "Construct a new diff based on the content above."
                    )
                },
            )
        except OSError as e:
            return ToolResult(
                success=False,
                error={"message": f"OS error updating {rel_path}: {e}"},
            )

    async def _delete_file(self, resolved: Path, rel_path: str) -> ToolResult:
        """Delete a file."""
        if not resolved.exists():
            return ToolResult(
                success=False,
                error={"message": f"File not found: {rel_path}"},
            )

        try:
            resolved.unlink()
            await self._emit_event("apply-patch:applied", rel_path, "delete")
            return ToolResult(success=True, output=f"D {rel_path}")
        except OSError as e:
            return ToolResult(
                success=False, error={"message": f"OS error deleting {rel_path}: {e}"}
            )

    def _resolve_path(self, rel_path: str) -> Path:
        """Resolve a relative path against the working directory."""
        path = Path(rel_path)
        if path.is_absolute():
            return path
        if self.working_dir:
            return Path(self.working_dir) / path
        return path

    def _validate_path(self, resolved: Path) -> tuple[bool, str | None]:
        """Validate path against allow/deny lists."""
        from ..path_validation import is_path_allowed

        return is_path_allowed(
            resolved, self.allowed_write_paths, self.denied_write_paths
        )

    async def _emit_event(self, event: str, path: str, operation: str) -> None:
        """Emit an observability event."""
        try:
            await self.coordinator.hooks.emit(
                event, {"path": path, "operation": operation}
            )
        except Exception:
            logger.debug(f"Failed to emit {event} for {path}", exc_info=True)
