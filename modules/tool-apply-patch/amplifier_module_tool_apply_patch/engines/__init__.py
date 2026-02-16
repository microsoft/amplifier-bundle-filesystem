"""Engine interface for the apply_patch tool.

Both engines (function and native) implement this protocol.
The tool shell delegates to whichever engine is configured.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from amplifier_core import ToolResult


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
