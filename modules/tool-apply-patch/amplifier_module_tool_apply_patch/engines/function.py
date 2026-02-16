"""Function engine for the apply_patch tool.

Receives a raw V4A patch string from the model, parses it into operations
(add / update / delete / rename), validates paths, and applies each operation
to the filesystem. Works with any provider.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from amplifier_core import ModuleCoordinator, ToolResult

from ..apply_diff import apply_diff

logger = logging.getLogger(__name__)

# V4A operation markers
_BEGIN_PATCH = "*** Begin Patch"
_END_PATCH = "*** End Patch"
_ADD_FILE = "*** Add File: "
_UPDATE_FILE = "*** Update File: "
_DELETE_FILE = "*** Delete File: "
_MOVE_TO = "*** Move to: "
_END_OF_FILE = "*** End of File"


@dataclass
class PatchOperation:
    """A single parsed operation from a V4A patch."""

    type: Literal["add", "update", "delete"]
    path: str
    diff: str = ""
    move_to: str | None = None


class FunctionEngine:
    """Function engine: parses V4A patches and applies them to the filesystem.

    Schema: {"patch": "<V4A patch string>"}
    Output: git-style summary ("M src/main.py\\nA src/new.py\\nD old.py")
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
            "Apply a patch to files using the V4A diff format. "
            "Supports adding, updating, deleting, and renaming files in a single patch. "
            "The patch must be wrapped in '*** Begin Patch' and '*** End Patch' markers."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "The patch content in V4A format",
                },
            },
            "required": ["patch"],
        }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute a V4A patch.

        Parses the patch into operations, validates all paths,
        then applies each operation in order.
        """
        patch_str = input.get("patch", "")
        if not patch_str:
            return ToolResult(success=False, error={"message": "patch is required"})

        # Parse the V4A patch into operations
        try:
            operations = self._parse_patch(patch_str)
        except ValueError as e:
            return ToolResult(
                success=False,
                error={"message": f"Patch parse error: {e}"},
            )

        if not operations:
            return ToolResult(
                success=False,
                error={"message": "Patch contains no operations"},
            )

        # Apply each operation
        summary_lines: list[str] = []
        try:
            for op in operations:
                status = await self._apply_operation(op)
                summary_lines.append(status)
        except _PatchError as e:
            return ToolResult(success=False, error={"message": str(e)})

        summary = "\n".join(summary_lines)
        return ToolResult(success=True, output=summary)

    def _parse_patch(self, patch_str: str) -> list[PatchOperation]:
        """Parse a V4A patch string into a list of operations."""
        lines = patch_str.split("\n")
        # Strip leading/trailing whitespace lines
        while lines and lines[0].strip() == "":
            lines.pop(0)
        while lines and lines[-1].strip() == "":
            lines.pop()

        # Expect *** Begin Patch header
        if not lines or not lines[0].rstrip().startswith(_BEGIN_PATCH):
            raise ValueError("Patch must start with '*** Begin Patch'")

        operations: list[PatchOperation] = []
        i = 1  # Skip *** Begin Patch

        while i < len(lines):
            line = lines[i].rstrip()

            if line.startswith(_END_PATCH):
                break

            if line.startswith(_ADD_FILE):
                path = line[len(_ADD_FILE) :]
                i += 1
                # Collect all +lines until next operation marker or end
                diff_lines: list[str] = []
                while i < len(lines):
                    raw_line = lines[i]
                    if raw_line.rstrip().startswith("*** "):
                        break
                    diff_lines.append(raw_line)
                    i += 1
                operations.append(
                    PatchOperation(type="add", path=path, diff="\n".join(diff_lines))
                )

            elif line.startswith(_UPDATE_FILE):
                path = line[len(_UPDATE_FILE) :]
                i += 1
                move_to: str | None = None
                # Check for *** Move to: marker
                if i < len(lines) and lines[i].rstrip().startswith(_MOVE_TO):
                    move_to = lines[i].rstrip()[len(_MOVE_TO) :]
                    i += 1
                # Collect diff lines until next operation marker or end
                diff_lines = []
                while i < len(lines):
                    raw_line = lines[i]
                    stripped = raw_line.rstrip()
                    if (
                        stripped.startswith(_END_PATCH)
                        or stripped.startswith(_ADD_FILE)
                        or stripped.startswith(_UPDATE_FILE)
                        or stripped.startswith(_DELETE_FILE)
                    ):
                        break
                    diff_lines.append(raw_line)
                    i += 1
                operations.append(
                    PatchOperation(
                        type="update",
                        path=path,
                        diff="\n".join(diff_lines),
                        move_to=move_to,
                    )
                )

            elif line.startswith(_DELETE_FILE):
                path = line[len(_DELETE_FILE) :]
                i += 1
                operations.append(PatchOperation(type="delete", path=path))

            else:
                raise ValueError(f"Unexpected line at position {i}: {line}")

        return operations

    async def _apply_operation(self, op: PatchOperation) -> str:
        """Apply a single parsed operation and return a git-style status line."""
        resolved = self._resolve_path(op.path)
        self._validate_path(resolved)

        if op.type == "add":
            return await self._apply_add(resolved, op)
        elif op.type == "update":
            return await self._apply_update(resolved, op)
        elif op.type == "delete":
            return await self._apply_delete(resolved, op)
        else:
            raise _PatchError(f"Unknown operation type: {op.type}")

    async def _apply_add(self, resolved: Path, op: PatchOperation) -> str:
        """Create a new file from +lines."""
        if resolved.exists():
            raise _PatchError(f"File already exists: {op.path}")

        # Create parent directories
        resolved.parent.mkdir(parents=True, exist_ok=True)

        # Use apply_diff in create mode to process +lines
        content = apply_diff("", op.diff, mode="create")
        resolved.write_text(content, encoding="utf-8")

        await self._emit_event("apply-patch:applied", op.path, "add")
        return f"A {op.path}"

    async def _apply_update(self, resolved: Path, op: PatchOperation) -> str:
        """Update an existing file with a diff."""
        if not resolved.exists():
            raise _PatchError(f"File not found: {op.path}")

        existing = resolved.read_text(encoding="utf-8")

        try:
            updated = apply_diff(existing, op.diff)
        except ValueError as e:
            raise _PatchError(
                f"Context mismatch in {op.path}: {e} — file may have changed. Read the file and retry."
            ) from e

        # Handle rename (Move to)
        if op.move_to:
            new_resolved = self._resolve_path(op.move_to)
            self._validate_path(new_resolved)
            new_resolved.parent.mkdir(parents=True, exist_ok=True)
            new_resolved.write_text(updated, encoding="utf-8")
            resolved.unlink()
            await self._emit_event("apply-patch:applied", op.path, "rename")
            return f"R {op.path} -> {op.move_to}"
        else:
            resolved.write_text(updated, encoding="utf-8")
            await self._emit_event("apply-patch:applied", op.path, "update")
            return f"M {op.path}"

    async def _apply_delete(self, resolved: Path, op: PatchOperation) -> str:
        """Delete a file."""
        if not resolved.exists():
            raise _PatchError(f"File not found: {op.path}")

        resolved.unlink()
        await self._emit_event("apply-patch:applied", op.path, "delete")
        return f"D {op.path}"

    def _resolve_path(self, rel_path: str) -> Path:
        """Resolve a relative path against the working directory."""
        path = Path(rel_path)
        if path.is_absolute():
            return path
        if self.working_dir:
            return Path(self.working_dir) / path
        return path

    def _validate_path(self, resolved: Path) -> None:
        """Validate path against allow/deny lists. Raises _PatchError on denial."""
        # Import shared validation
        bundle_shared = str(
            Path(__file__).resolve().parent.parent.parent.parent.parent / "shared"
        )
        if bundle_shared not in sys.path:
            sys.path.insert(0, bundle_shared)

        from filesystem_utils.path_validation import is_path_allowed  # type: ignore[import-not-found]

        allowed, error_msg = is_path_allowed(
            resolved, self.allowed_write_paths, self.denied_write_paths
        )
        if not allowed:
            raise _PatchError(error_msg or "Access denied")

    async def _emit_event(self, event: str, path: str, operation: str) -> None:
        """Emit an observability event."""
        try:
            await self.coordinator.hooks.emit(
                event, {"path": path, "operation": operation}
            )
        except Exception:
            # Event emission should never break the tool
            logger.debug(f"Failed to emit {event} for {path}", exc_info=True)


class _PatchError(Exception):
    """Internal error for patch application failures."""

    pass
