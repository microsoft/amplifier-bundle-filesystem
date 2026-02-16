"""ApplyPatchTool — the tool shell that delegates to engines.

This class satisfies the Amplifier Tool protocol:
  - name (str)
  - description (str)
  - input_schema (dict)
  - execute(input) -> ToolResult

It selects and delegates to whichever engine is configured ("native" or "function").
Each engine provides its own description, schema, and execute implementation.
"""

from __future__ import annotations

from typing import Any

from amplifier_core import ModuleCoordinator, ToolResult

from .engines.function import FunctionEngine
from .engines.native import NativeEngine


class ApplyPatchTool:
    """Apply patches to files using the V4A diff format.

    Delegates to one of two engines based on config:
    - "native": Pre-parsed operations from the OpenAI Responses API
    - "function": Raw V4A patch strings, works with any provider
    """

    name = "apply_patch"

    def __init__(self, config: dict[str, Any], coordinator: ModuleCoordinator) -> None:
        self.config = config
        self.coordinator = coordinator

        engine_name = config.get("engine", "native")
        if engine_name == "native":
            self._engine = NativeEngine(config, coordinator)
        elif engine_name == "function":
            self._engine = FunctionEngine(config, coordinator)
        else:
            raise ValueError(
                f"Unknown engine: {engine_name!r}. Must be 'native' or 'function'."
            )

    @property
    def description(self) -> str:
        """Delegate description to the active engine."""
        return self._engine.description

    @property
    def input_schema(self) -> dict[str, Any]:
        """Delegate schema to the active engine."""
        return self._engine.input_schema

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Delegate execution to the active engine."""
        return await self._engine.execute(input)  # type: ignore[return-value]
