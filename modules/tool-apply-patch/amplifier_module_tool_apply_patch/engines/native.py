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
        if resolved.exists():
            return ToolResult(
                success=False,
                error={
                    "message": (
                        f"File already exists: {rel_path}. "
                        "Use update_file to modify existing files."
                    )
                },
            )

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

        try:
            existing = resolved.read_text(encoding="utf-8")
            updated = apply_diff(existing, diff)
            resolved.write_text(updated, encoding="utf-8")
            await self._emit_event("apply-patch:applied", rel_path, "update")
            return ToolResult(success=True, output=f"M {rel_path}")
        except ValueError as e:
            return ToolResult(
                success=False,
                error={
                    "message": f"Context mismatch in {rel_path}: {e} — file may have changed. Read the file and retry."
                },
            )
        except OSError as e:
            return ToolResult(
                success=False, error={"message": f"OS error updating {rel_path}: {e}"}
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
