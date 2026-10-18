"""Tests for ApplyPatchTool class — config routing and engine selection."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from amplifier_core import ModuleCoordinator
from amplifier_module_tool_apply_patch import mount
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


@pytest_asyncio.fixture
async def mount_patch():
    """Mount the actual module and engines on real Core coordinators."""
    coordinators = []

    async def mount_at(directory, config):
        coordinator = ModuleCoordinator()
        coordinators.append(coordinator)
        coordinator.register_capability("session.working_dir", str(directory))
        await mount(coordinator, config)
        tool = coordinator.get("tools")["apply_patch"]
        assert isinstance(tool, ApplyPatchTool)
        return tool

    yield mount_at
    for coordinator in reversed(coordinators):
        try:
            await coordinator.unmount("tools", name="apply_patch")
        finally:
            await coordinator.cleanup()


async def _mounted_update(tool, engine, path="file.txt"):
    if engine == "native":
        return await tool.execute(
            {"type": "update_file", "path": path, "diff": "@@\n-old\n+new"}
        )
    marker = "***"
    return await tool.execute(
        {
            "patch": (
                f"{marker} Begin Patch\n{marker} Update File: {path}\n"
                f"@@\n-old\n+new\n{marker} End Patch"
            )
        }
    )


async def _assert_mounted_update(tool, engine):
    result = await _mounted_update(tool, engine)
    assert result.success is True
    assert result.output == "M file.txt"
    assert result.error is None


@pytest.mark.asyncio
class TestMountWorkingDirectory:
    @pytest.mark.parametrize("engine", ["native", "function"])
    async def test_nonempty_declaration_is_not_mutated(
        self, tmp_path, mount_patch, engine
    ):
        config = {
            "engine": engine,
            "allowed_write_paths": [str(tmp_path)],
            "denied_write_paths": [str(tmp_path / "private")],
        }
        original = deepcopy(config)
        (tmp_path / "file.txt").write_text("old\n")

        tool = await mount_patch(tmp_path, config)
        assert tool._engine.working_dir == str(tmp_path)
        await _assert_mounted_update(tool, engine)
        assert (tmp_path / "file.txt").read_text() == "new\n"
        assert config == original
        assert "working_dir" not in config

    @pytest.mark.parametrize("engine", ["native", "function"])
    @pytest.mark.parametrize("serialized", [False, True], ids=["reuse", "json-roundtrip"])
    async def test_declaration_uses_each_session_directory(
        self, tmp_path, mount_patch, engine, serialized
    ):
        """A JSON roundtrip is a module boundary check, not a host handoff."""
        first, second = tmp_path / "first", tmp_path / "second"
        for directory in (first, second):
            directory.mkdir()
            (directory / "file.txt").write_text("old\n")
        config = {
            "engine": engine,
            "allowed_write_paths": [str(first), str(second)],
        }
        original = deepcopy(config)
        first_tool = await mount_patch(first, config)
        next_config = json.loads(json.dumps(config)) if serialized else config
        second_tool = await mount_patch(second, next_config)

        assert first_tool._engine.working_dir == str(first)
        assert second_tool._engine.working_dir == str(second)
        await _assert_mounted_update(second_tool, engine)
        assert (second / "file.txt").read_text() == "new\n"
        assert (first / "file.txt").read_text() == "old\n"
        await _assert_mounted_update(first_tool, engine)
        assert (first / "file.txt").read_text() == "new\n"
        assert config == next_config == original

    @pytest.mark.parametrize("engine", ["native", "function"])
    @pytest.mark.parametrize("pin_name", ["first", "explicit"])
    async def test_explicit_pin_survives_a_new_session(
        self, tmp_path, mount_patch, engine, pin_name
    ):
        first, second, explicit = (
            tmp_path / name for name in ("first", "second", "explicit")
        )
        for directory in (first, second, explicit):
            directory.mkdir()
            (directory / "file.txt").write_text("old\n")
        pinned = tmp_path / pin_name
        config = {
            "engine": engine,
            "working_dir": str(pinned),
            "allowed_write_paths": [str(pinned)],
        }
        original = deepcopy(config)
        first_tool = await mount_patch(first, config)
        second_tool = await mount_patch(second, json.loads(json.dumps(config)))

        assert first_tool._engine.working_dir == str(pinned)
        assert second_tool._engine.working_dir == str(pinned)
        await _assert_mounted_update(second_tool, engine)
        assert (pinned / "file.txt").read_text() == "new\n"
        assert (second / "file.txt").read_text() == "old\n"
        assert config == original

    @pytest.mark.parametrize("engine", ["native", "function"])
    async def test_old_pin_outside_new_allowlist_remains_denied(
        self, tmp_path, mount_patch, engine
    ):
        """An ambiguous saved pin must not be cleared or retargeted implicitly."""
        old, current = tmp_path / "old", tmp_path / "current"
        for directory in (old, current):
            directory.mkdir()
            (directory / "file.txt").write_text("old\n")
        config = {
            "engine": engine,
            "working_dir": str(old),
            "allowed_write_paths": [str(current)],
        }
        original = deepcopy(config)
        tool = await mount_patch(current, config)

        result = await _mounted_update(tool, engine)
        assert not result.success
        assert tool._engine.working_dir == str(old)
        assert (old / "file.txt").read_text() == "old\n"
        assert (current / "file.txt").read_text() == "old\n"
        assert config == original

    @pytest.mark.parametrize("engine", ["native", "function"])
    @pytest.mark.parametrize("pin", [None, ""])
    async def test_present_falsey_pin_keeps_existing_semantics(
        self, tmp_path, mount_patch, engine, pin
    ):
        config = {"engine": engine, "working_dir": pin}
        original = deepcopy(config)
        tool = await mount_patch(tmp_path, config)
        assert tool._engine.working_dir == pin
        assert config == original

    async def test_empty_declaration_control(self, tmp_path, mount_patch):
        config = {}
        tool = await mount_patch(tmp_path, config)
        assert isinstance(tool._engine, NativeEngine)
        assert tool._engine.working_dir == str(tmp_path)
        assert config == {}
