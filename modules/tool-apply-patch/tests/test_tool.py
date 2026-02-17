"""Tests for ApplyPatchTool class — config routing and engine selection."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_tool_apply_patch.tool import ApplyPatchTool
from amplifier_module_tool_apply_patch.engines.function import FunctionEngine
from amplifier_module_tool_apply_patch.engines.native import NativeEngine


def _make_tool(tmp_path: Path, engine: str = "native") -> ApplyPatchTool:
    config: dict[str, Any] = {
        "engine": engine,
        "working_dir": str(tmp_path),
        "allowed_write_paths": [str(tmp_path)],
        "denied_write_paths": [],
    }
    coordinator = MagicMock()
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock()
    coordinator.get_capability = MagicMock(return_value=None)
    return ApplyPatchTool(config, coordinator)


class TestToolAttributes:
    def test_name_is_apply_patch(self, tmp_path: Path) -> None:
        tool = _make_tool(tmp_path)
        assert tool.name == "apply_patch"

    def test_has_description(self, tmp_path: Path) -> None:
        tool = _make_tool(tmp_path)
        assert isinstance(tool.description, str)
        assert len(tool.description) > 0

    def test_has_input_schema(self, tmp_path: Path) -> None:
        tool = _make_tool(tmp_path)
        schema = tool.input_schema
        assert isinstance(schema, dict)
        assert schema["type"] == "object"


class TestEngineSelection:
    def test_default_engine_is_native(self, tmp_path: Path) -> None:
        """When no engine is specified in config, default to native."""
        config: dict[str, Any] = {
            "working_dir": str(tmp_path),
            "allowed_write_paths": [str(tmp_path)],
            "denied_write_paths": [],
        }
        coordinator = MagicMock()
        coordinator.hooks = MagicMock()
        coordinator.hooks.emit = AsyncMock()
        coordinator.get_capability = MagicMock(return_value=None)
        tool = ApplyPatchTool(config, coordinator)
        assert isinstance(tool._engine, NativeEngine)

    def test_native_engine_selected(self, tmp_path: Path) -> None:
        tool = _make_tool(tmp_path, engine="native")
        assert isinstance(tool._engine, NativeEngine)

    def test_function_engine_selected(self, tmp_path: Path) -> None:
        tool = _make_tool(tmp_path, engine="function")
        assert isinstance(tool._engine, FunctionEngine)

    def test_unknown_engine_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unknown engine"):
            _make_tool(tmp_path, engine="rust")


class TestToolDelegation:
    @pytest.mark.asyncio
    async def test_execute_delegates_to_engine(self, tmp_path: Path) -> None:
        """Tool.execute() should delegate to the engine's execute()."""
        tool = _make_tool(tmp_path, engine="native")
        (tmp_path / "test.py").write_text("old\n")

        result = await tool.execute(
            {
                "type": "update_file",
                "path": "test.py",
                "diff": "@@\n-old\n+new",
            }
        )
        assert result.success is True
        assert (tmp_path / "test.py").read_text() == "new\n"

    @pytest.mark.asyncio
    async def test_function_engine_execute(self, tmp_path: Path) -> None:
        """Function engine receives patches via tool shell."""
        tool = _make_tool(tmp_path, engine="function")
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Add File: hello.py",
                "+print('hi')",
                "*** End Patch",
            ]
        )
        result = await tool.execute({"patch": patch})
        assert result.success is True
        assert (tmp_path / "hello.py").exists()

    def test_get_schema_returns_engine_schema(self, tmp_path: Path) -> None:
        """get_schema() should return the engine's input_schema."""
        tool = _make_tool(tmp_path, engine="native")
        schema = tool.input_schema
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        # Native engine schema has "type" and "path" properties
        assert "type" in schema["properties"]
        assert "path" in schema["properties"]

    def test_function_engine_schema_differs(self, tmp_path: Path) -> None:
        """Function engine has a different schema (patch property)."""
        tool = _make_tool(tmp_path, engine="function")
        schema = tool.input_schema
        assert isinstance(schema, dict)
        assert "patch" in schema["properties"]
