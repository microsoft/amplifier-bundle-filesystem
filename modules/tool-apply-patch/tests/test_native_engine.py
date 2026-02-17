"""Tests for the native engine.

The native engine receives pre-parsed operations from the OpenAI provider.
Each call has {type, path, diff} — no V4A parsing needed (the API already
decomposed the patch into individual operations).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from amplifier_module_tool_apply_patch.engines.native import NativeEngine


def _make_engine(tmp_path: Path) -> NativeEngine:
    """Create a NativeEngine configured to write only within tmp_path."""
    config: dict[str, Any] = {
        "working_dir": str(tmp_path),
        "allowed_write_paths": [str(tmp_path)],
        "denied_write_paths": [],
    }
    coordinator = MagicMock()
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock()
    return NativeEngine(config, coordinator)


class TestNativeEngineSchema:
    def test_has_description(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        assert isinstance(engine.description, str)
        assert len(engine.description) > 0

    def test_has_input_schema(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        schema = engine.input_schema
        assert schema["type"] == "object"
        assert "type" in schema["properties"]
        assert "path" in schema["properties"]


class TestNativeEngineCreateFile:
    @pytest.mark.asyncio
    async def test_create_file(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "new.py",
                "diff": "+print('hello')\n+print('world')",
            }
        )
        assert result.success is True
        created = tmp_path / "new.py"
        assert created.exists()
        content = created.read_text()
        assert "print('hello')" in content

    @pytest.mark.asyncio
    async def test_create_file_in_subdirectory(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "src/lib/utils.py",
                "diff": "+def util():\n+    pass",
            }
        )
        assert result.success is True
        assert (tmp_path / "src" / "lib" / "utils.py").exists()


class TestNativeEngineUpdateFile:
    @pytest.mark.asyncio
    async def test_update_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "main.py"
        target.write_text("line1\nold_line\nline3\n")

        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "update_file",
                "path": "main.py",
                "diff": "@@ line1\n-old_line\n+new_line\n line3",
            }
        )
        assert result.success is True
        assert target.read_text() == "line1\nnew_line\nline3\n"

    @pytest.mark.asyncio
    async def test_update_nonexistent_file_fails(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "update_file",
                "path": "missing.py",
                "diff": "@@\n-x\n+y",
            }
        )
        assert result.success is False
        assert "not found" in result.error["message"].lower()


class TestNativeEngineDeleteFile:
    @pytest.mark.asyncio
    async def test_delete_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "old.py"
        target.write_text("bye")

        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "delete_file",
                "path": "old.py",
            }
        )
        assert result.success is True
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_file_fails(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "delete_file",
                "path": "ghost.py",
            }
        )
        assert result.success is False


class TestNativeEngineErrors:
    @pytest.mark.asyncio
    async def test_missing_type_returns_error(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = await engine.execute({"path": "file.py"})
        assert result.success is False
        assert "type" in result.error["message"].lower()

    @pytest.mark.asyncio
    async def test_missing_path_returns_error(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = await engine.execute({"type": "create_file"})
        assert result.success is False
        assert "path" in result.error["message"].lower()

    @pytest.mark.asyncio
    async def test_unknown_type_returns_error(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "rename_file",
                "path": "file.py",
            }
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_path_outside_allowed_dirs_is_denied(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "/etc/evil.py",
                "diff": "+bad",
            }
        )
        assert result.success is False

    @pytest.mark.asyncio
    async def test_context_mismatch_returns_error(self, tmp_path: Path) -> None:
        (tmp_path / "file.py").write_text("actual\n")
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "update_file",
                "path": "file.py",
                "diff": "@@ wrong_context\n-nope\n+yes",
            }
        )
        assert result.success is False
