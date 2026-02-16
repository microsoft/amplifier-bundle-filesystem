"""Tests for the function engine.

The function engine receives a raw V4A patch string, parses it into
operations (add/update/delete/rename), and applies them against the filesystem.
It works with any provider — the model just calls apply_patch(patch="...").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

# We need sys.path for the shared utils
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent.parent.parent / "shared"),
)

from amplifier_module_tool_apply_patch.engines.function import FunctionEngine


def _make_engine(tmp_path: Path) -> FunctionEngine:
    """Create a FunctionEngine configured to write only within tmp_path."""
    config: dict[str, Any] = {
        "working_dir": str(tmp_path),
        "allowed_write_paths": [str(tmp_path)],
        "denied_write_paths": [],
    }
    coordinator = MagicMock()
    coordinator.hooks = MagicMock()
    coordinator.hooks.emit = AsyncMock()
    return FunctionEngine(config, coordinator)


class TestFunctionEngineDescription:
    def test_has_description(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        assert isinstance(engine.description, str)
        assert len(engine.description) > 0

    def test_has_input_schema(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        schema = engine.input_schema
        assert schema["type"] == "object"
        assert "patch" in schema["properties"]
        assert "patch" in schema["required"]


class TestFunctionEngineAddFile:
    @pytest.mark.asyncio
    async def test_add_single_file(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Add File: hello.py",
                "+print('hello')",
                "+",
                "*** End Patch",
            ]
        )
        result = await engine.execute({"patch": patch})
        assert result.success is True
        assert (tmp_path / "hello.py").read_text() == "print('hello')\n"

    @pytest.mark.asyncio
    async def test_add_file_in_subdirectory(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Add File: src/utils/helper.py",
                "+def helper():",
                "+    return 42",
                "*** End Patch",
            ]
        )
        result = await engine.execute({"patch": patch})
        assert result.success is True
        assert (tmp_path / "src" / "utils" / "helper.py").exists()
        content = (tmp_path / "src" / "utils" / "helper.py").read_text()
        assert "def helper():" in content

    @pytest.mark.asyncio
    async def test_add_file_output_contains_status(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Add File: new.py",
                "+x = 1",
                "*** End Patch",
            ]
        )
        result = await engine.execute({"patch": patch})
        assert result.success is True
        # Output should contain git-style status
        assert "A" in str(result.output)
        assert "new.py" in str(result.output)


class TestFunctionEngineUpdateFile:
    @pytest.mark.asyncio
    async def test_update_existing_file(self, tmp_path: Path) -> None:
        # Setup: create the file to be updated
        target = tmp_path / "main.py"
        target.write_text("line1\nline2\nline3\n")

        engine = _make_engine(tmp_path)
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Update File: main.py",
                "@@ line1",
                "-line2",
                "+updated_line2",
                " line3",
                "*** End Patch",
            ]
        )
        result = await engine.execute({"patch": patch})
        assert result.success is True
        assert target.read_text() == "line1\nupdated_line2\nline3\n"

    @pytest.mark.asyncio
    async def test_update_nonexistent_file_fails(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Update File: missing.py",
                "@@ line1",
                "-line2",
                "+updated",
                "*** End Patch",
            ]
        )
        result = await engine.execute({"patch": patch})
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error["message"].lower()

    @pytest.mark.asyncio
    async def test_update_output_contains_status(self, tmp_path: Path) -> None:
        target = tmp_path / "app.py"
        target.write_text("old\n")

        engine = _make_engine(tmp_path)
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Update File: app.py",
                "@@",
                "-old",
                "+new",
                "*** End Patch",
            ]
        )
        result = await engine.execute({"patch": patch})
        assert result.success is True
        assert "M" in str(result.output)
        assert "app.py" in str(result.output)


class TestFunctionEngineDeleteFile:
    @pytest.mark.asyncio
    async def test_delete_existing_file(self, tmp_path: Path) -> None:
        target = tmp_path / "obsolete.py"
        target.write_text("old code")

        engine = _make_engine(tmp_path)
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Delete File: obsolete.py",
                "*** End Patch",
            ]
        )
        result = await engine.execute({"patch": patch})
        assert result.success is True
        assert not target.exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_file_fails(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Delete File: ghost.py",
                "*** End Patch",
            ]
        )
        result = await engine.execute({"patch": patch})
        assert result.success is False
        assert result.error is not None
        assert "not found" in result.error["message"].lower()

    @pytest.mark.asyncio
    async def test_delete_output_contains_status(self, tmp_path: Path) -> None:
        target = tmp_path / "dead.py"
        target.write_text("bye")

        engine = _make_engine(tmp_path)
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Delete File: dead.py",
                "*** End Patch",
            ]
        )
        result = await engine.execute({"patch": patch})
        assert result.success is True
        assert "D" in str(result.output)
        assert "dead.py" in str(result.output)


class TestFunctionEngineMultiFile:
    @pytest.mark.asyncio
    async def test_multi_file_patch(self, tmp_path: Path) -> None:
        """A single patch that adds, updates, and deletes files."""
        # Setup existing files
        (tmp_path / "existing.py").write_text("old_code\n")
        (tmp_path / "to_delete.py").write_text("bye")

        engine = _make_engine(tmp_path)
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Add File: new_file.py",
                "+print('new')",
                "*** Update File: existing.py",
                "@@",
                "-old_code",
                "+new_code",
                "*** Delete File: to_delete.py",
                "*** End Patch",
            ]
        )
        result = await engine.execute({"patch": patch})
        assert result.success is True
        assert (tmp_path / "new_file.py").exists()
        assert (tmp_path / "existing.py").read_text() == "new_code\n"
        assert not (tmp_path / "to_delete.py").exists()


class TestFunctionEngineRename:
    @pytest.mark.asyncio
    async def test_rename_with_update(self, tmp_path: Path) -> None:
        """Update File + Move to = rename with modifications."""
        (tmp_path / "old_name.py").write_text("import os\nimport old_dep\n")

        engine = _make_engine(tmp_path)
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Update File: old_name.py",
                "*** Move to: new_name.py",
                "@@ import os",
                "-import old_dep",
                "+import new_dep",
                "*** End Patch",
            ]
        )
        result = await engine.execute({"patch": patch})
        assert result.success is True
        assert not (tmp_path / "old_name.py").exists()
        assert (tmp_path / "new_name.py").exists()
        content = (tmp_path / "new_name.py").read_text()
        assert "import new_dep" in content
        assert "import old_dep" not in content


class TestFunctionEngineErrors:
    @pytest.mark.asyncio
    async def test_malformed_patch_returns_error(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = await engine.execute({"patch": "not a valid patch at all"})
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_missing_patch_key_returns_error(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        result = await engine.execute({})
        assert result.success is False
        assert result.error is not None
        assert "patch" in result.error["message"].lower()

    @pytest.mark.asyncio
    async def test_path_outside_allowed_dirs_is_denied(self, tmp_path: Path) -> None:
        engine = _make_engine(tmp_path)
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Add File: /etc/evil.py",
                "+bad",
                "*** End Patch",
            ]
        )
        result = await engine.execute({"patch": patch})
        assert result.success is False
        assert result.error is not None
        assert (
            "denied" in result.error["message"].lower()
            or "allowed" in result.error["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_context_mismatch_returns_error(self, tmp_path: Path) -> None:
        (tmp_path / "file.py").write_text("actual_content\n")
        engine = _make_engine(tmp_path)
        patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Update File: file.py",
                "@@ wrong_context",
                "-wrong_line",
                "+replacement",
                "*** End Patch",
            ]
        )
        result = await engine.execute({"patch": patch})
        assert result.success is False
