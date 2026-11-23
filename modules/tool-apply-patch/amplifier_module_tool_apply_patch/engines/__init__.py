"""Engine interface for the apply_patch tool.

Both engines (function and native) implement this protocol.
The tool shell delegates to whichever engine is configured.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from amplifier_core import ModuleCoordinator, ToolResult


@runtime_checkable
class ApplyPatchEngine(Protocol):
    """Protocol for apply_patch engine implementations."""

    @property
    def description(self) -> str:
        """Tool description shown to the model."""
        ...

    @property
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's input parameters."""
        ...

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Execute a patch operation.

        Args:
            input: Tool input matching input_schema.

        Returns:
            ToolResult with success/failure and output/error.
        """
        ...


async def emit_applied(
    coordinator: ModuleCoordinator, path: str, operation: str, written_path: Path | None = None
) -> None:
    """Publish a completed effect without changing the patch result."""
    try:
        data = {"path": path, "operation": operation}
        if written_path is not None:
            data["written_path"] = str(written_path.resolve())
        await coordinator.hooks.emit("apply-patch:applied", data)
    except Exception:
        logging.getLogger(__name__).debug("Could not report patch effect for %s", path, exc_info=True)
