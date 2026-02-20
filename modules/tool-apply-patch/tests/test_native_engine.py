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

    @pytest.mark.asyncio
    async def test_create_file_already_exists_returns_error(
        self, tmp_path: Path
    ) -> None:
        """create_file on an existing file should fail, not silently overwrite."""
        existing = tmp_path / "already.py"
        existing.write_text("original content\n")

        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "already.py",
                "diff": "+overwritten content\n",
            }
        )
        assert result.success is False
        assert "already exists" in result.error["message"].lower()
        # File content must be unchanged — no silent overwrite
        assert existing.read_text() == "original content\n"

    @pytest.mark.asyncio
    async def test_create_file_already_exists_suggests_update(
        self, tmp_path: Path
    ) -> None:
        """Error message should tell the model to use update_file instead."""
        existing = tmp_path / "config.yaml"
        existing.write_text("key: value\n")

        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "config.yaml",
                "diff": "+key: new_value\n",
            }
        )
        assert result.success is False
        assert "update_file" in result.error["message"]


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


class TestNativeEngineSelfHealingErrors:
    """Error responses include current file content so models can self-correct."""

    @pytest.mark.asyncio
    async def test_context_mismatch_includes_current_content(
        self, tmp_path: Path
    ) -> None:
        """Context mismatch error should include the file's actual content."""
        (tmp_path / "file.py").write_text("actual_line_one\nactual_line_two\n")
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "update_file",
                "path": "file.py",
                "diff": "@@ wrong_context\n-nope\n+yes",
            }
        )
        assert result.success is False
        msg = result.error["message"]
        # Must include the actual file content so the model can construct a correct diff
        assert "actual_line_one" in msg
        assert "actual_line_two" in msg
        # Should NOT say "Read the file and retry" — we already gave the content
        assert "Read the file" not in msg
        assert "Construct a new diff" in msg

    @pytest.mark.asyncio
    async def test_context_mismatch_includes_line_numbers(self, tmp_path: Path) -> None:
        """Content hint should include line numbers for easier diff construction."""
        (tmp_path / "file.py").write_text("alpha\nbeta\ngamma\n")
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "update_file",
                "path": "file.py",
                "diff": "@@ wrong\n-nope\n+yes",
            }
        )
        assert result.success is False
        msg = result.error["message"]
        assert "1\talpha" in msg
        assert "2\tbeta" in msg
        assert "3\tgamma" in msg

    @pytest.mark.asyncio
    async def test_file_already_exists_includes_current_content(
        self, tmp_path: Path
    ) -> None:
        """File-already-exists error should include the file's current content."""
        (tmp_path / "existing.py").write_text("original_content\n")
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "existing.py",
                "diff": "+new_content\n",
            }
        )
        assert result.success is False
        msg = result.error["message"]
        assert "already exists" in msg.lower()
        assert "update_file" in msg
        # Must include current file content
        assert "original_content" in msg

    @pytest.mark.asyncio
    async def test_empty_file_context_mismatch_shows_empty_hint(
        self, tmp_path: Path
    ) -> None:
        """Empty files should get a clear 'empty' hint, not crash."""
        (tmp_path / "empty.py").write_text("")
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "update_file",
                "path": "empty.py",
                "diff": "@@ something\n-nope\n+yes",
            }
        )
        assert result.success is False
        msg = result.error["message"]
        assert "empty" in msg.lower() or "0 lines" in msg


class TestNativeEngineV4ADetection:
    """Native engine should detect V4A wrapper markers and return actionable errors."""

    @pytest.mark.asyncio
    async def test_begin_patch_in_diff_returns_actionable_error(
        self, tmp_path: Path
    ) -> None:
        """When diff contains '*** Begin Patch', error should mention V4A wrapper markers."""
        (tmp_path / "file.py").write_text("hello world\n")
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "update_file",
                "path": "file.py",
                "diff": "*** Begin Patch\n*** Update File: file.py\n@@ hello world\n-hello world\n+hello universe\n*** End Patch",
            }
        )
        assert result.success is False
        # Must mention V4A wrapper markers — not "Context mismatch" or "file may have changed"
        assert (
            "V4A" in result.error["message"]
            or "wrapper" in result.error["message"].lower()
        )
        assert "file may have changed" not in result.error["message"]

    @pytest.mark.asyncio
    async def test_update_file_marker_in_diff_returns_actionable_error(
        self, tmp_path: Path
    ) -> None:
        """When diff contains '*** Update File:', error should be actionable."""
        (tmp_path / "file.py").write_text("hello\n")
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "update_file",
                "path": "file.py",
                "diff": "*** Update File: file.py\n@@\n-hello\n+goodbye",
            }
        )
        assert result.success is False
        assert (
            "V4A" in result.error["message"]
            or "wrapper" in result.error["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_add_file_marker_in_diff_returns_actionable_error(
        self, tmp_path: Path
    ) -> None:
        """When diff contains '*** Add File:', error should be actionable."""
        (tmp_path / "file.py").write_text("hello\n")
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "update_file",
                "path": "file.py",
                "diff": "*** Add File: file.py\n+hello\n+goodbye",
            }
        )
        assert result.success is False
        assert (
            "V4A" in result.error["message"]
            or "wrapper" in result.error["message"].lower()
        )

    @pytest.mark.asyncio
    async def test_begin_patch_in_create_diff_returns_actionable_error(
        self, tmp_path: Path
    ) -> None:
        """V4A detection should also work for create_file operations."""
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "new.py",
                "diff": "*** Begin Patch\n*** Add File: new.py\n+print('hello')\n*** End Patch",
            }
        )
        assert result.success is False
        assert (
            "V4A" in result.error["message"]
            or "wrapper" in result.error["message"].lower()
        )
        assert "file may have changed" not in result.error["message"]

    @pytest.mark.asyncio
    async def test_clean_diff_still_works(self, tmp_path: Path) -> None:
        """Clean raw hunks (no V4A wrapper) should still succeed."""
        (tmp_path / "file.py").write_text("hello\n")
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "update_file",
                "path": "file.py",
                "diff": "@@\n-hello\n+goodbye",
            }
        )
        assert result.success is True
        assert (tmp_path / "file.py").read_text() == "goodbye\n"


class TestNativeEngineUnifiedDiffCreate:
    """Unified diff normalization: create mode."""

    @pytest.mark.asyncio
    async def test_unified_diff_headers_stripped_on_create(
        self, tmp_path: Path
    ) -> None:
        """Standard unified diff headers (---, +++, @@) are stripped on create."""
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "hello.py",
                "diff": (
                    "--- /dev/null\n"
                    "+++ b/hello.py\n"
                    "@@ -0,0 +1,2 @@\n"
                    "+print('hello')\n"
                    "+print('world')"
                ),
            }
        )
        assert result.success is True
        content = (tmp_path / "hello.py").read_text()
        assert content == "print('hello')\nprint('world')"

    @pytest.mark.asyncio
    async def test_diff_git_preamble_stripped_on_create(self, tmp_path: Path) -> None:
        """diff --git preamble followed by standard headers is stripped."""
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "hello.py",
                "diff": (
                    "diff --git a/hello.py b/hello.py\n"
                    "--- /dev/null\n"
                    "+++ b/hello.py\n"
                    "@@ -0,0 +1,2 @@\n"
                    "+print('hello')\n"
                    "+print('world')"
                ),
            }
        )
        assert result.success is True
        content = (tmp_path / "hello.py").read_text()
        assert content == "print('hello')\nprint('world')"

    @pytest.mark.asyncio
    async def test_valid_v4a_create_passes_through_unchanged(
        self, tmp_path: Path
    ) -> None:
        """Already-valid V4A create diff (only + lines) is not modified."""
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "hello.py",
                "diff": "+print('hello')\n+print('world')",
            }
        )
        assert result.success is True
        content = (tmp_path / "hello.py").read_text()
        assert content == "print('hello')\nprint('world')"

    @pytest.mark.asyncio
    async def test_no_newline_marker_stripped_on_create(self, tmp_path: Path) -> None:
        """\\ No newline at end of file marker is stripped."""
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "hello.py",
                "diff": (
                    "--- /dev/null\n"
                    "+++ b/hello.py\n"
                    "@@ -0,0 +1,2 @@\n"
                    "+print('hello')\n"
                    "+print('world')\n"
                    "\\ No newline at end of file"
                ),
            }
        )
        assert result.success is True
        content = (tmp_path / "hello.py").read_text()
        assert content == "print('hello')\nprint('world')"

    @pytest.mark.asyncio
    async def test_create_file_with_raw_content_no_plus_prefixes(
        self, tmp_path: Path
    ) -> None:
        """Raw content without '+' prefixes is normalized and file is created.

        This is the exact scenario from the Responses API bug: the API sends
        bare markdown/code content, not unified-diff-formatted '+' lines.
        Empty lines in the content must also be handled correctly.
        """
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "readme.md",
                "diff": "# Title\n\nSome body text\nMore content",
            }
        )
        assert result.success is True
        content = (tmp_path / "readme.md").read_text()
        assert content == "# Title\n\nSome body text\nMore content"

    @pytest.mark.asyncio
    async def test_create_file_with_mixed_prefixes(self, tmp_path: Path) -> None:
        """Content where some lines have '+' prefix and some don't.

        Lines already prefixed with '+' are left alone. Lines without '+'
        get the prefix added. This handles the edge case where bare content
        happens to have lines starting with '+'.
        """
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "mixed.py",
                "diff": "+print('hello')\nbare line\n+print('world')",
            }
        )
        assert result.success is True
        content = (tmp_path / "mixed.py").read_text()
        assert content == "print('hello')\nbare line\nprint('world')"


class TestNativeEngineUnifiedDiffUpdate:
    """Unified diff normalization: update mode."""

    @pytest.mark.asyncio
    async def test_unified_diff_headers_stripped_on_update(
        self, tmp_path: Path
    ) -> None:
        """Unified diff with --- a/path, +++ b/path, numeric @@ is normalized for update."""
        target = tmp_path / "main.py"
        target.write_text("line1\nold_line\nline3\n")

        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "update_file",
                "path": "main.py",
                "diff": (
                    "--- a/main.py\n"
                    "+++ b/main.py\n"
                    "@@ -1,3 +1,3 @@\n"
                    " line1\n"
                    "-old_line\n"
                    "+new_line\n"
                    " line3"
                ),
            }
        )
        assert result.success is True
        assert target.read_text() == "line1\nnew_line\nline3\n"

    @pytest.mark.asyncio
    async def test_v4a_text_anchor_preserved_on_update(self, tmp_path: Path) -> None:
        """V4A text anchor (@@ def hello():) must NOT be stripped — critical no-regression."""
        target = tmp_path / "main.py"
        target.write_text("def hello():\n    old_line\n    other_line\n")

        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "update_file",
                "path": "main.py",
                "diff": "@@ def hello():\n-    old_line\n+    new_line",
            }
        )
        assert result.success is True
        assert target.read_text() == "def hello():\n    new_line\n    other_line\n"

    @pytest.mark.asyncio
    async def test_numeric_hunk_header_with_trailing_context_stripped(
        self, tmp_path: Path
    ) -> None:
        """Numeric @@ -N,M +N,M @@ with trailing function name is stripped."""
        target = tmp_path / "main.py"
        target.write_text("def hello():\n    old_line\n    other_line\n")

        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "update_file",
                "path": "main.py",
                "diff": (
                    "@@ -1,3 +1,3 @@ def hello():\n"
                    "-    old_line\n"
                    "+    new_line\n"
                    "     other_line"
                ),
            }
        )
        assert result.success is True
        assert target.read_text() == "def hello():\n    new_line\n    other_line\n"


class TestNativeEngineUnifiedDiffEdgeCases:
    """Unified diff normalization: edge cases."""

    @pytest.mark.asyncio
    async def test_empty_diff_passes_through(self, tmp_path: Path) -> None:
        """Empty diff passes through the normalizer unchanged."""
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "empty.py",
                "diff": "",
            }
        )
        # Empty diff in create mode creates an empty file
        assert result.success is True
        assert (tmp_path / "empty.py").read_text() == ""

    @pytest.mark.asyncio
    async def test_headers_only_no_content_creates_empty_file(
        self, tmp_path: Path
    ) -> None:
        """Diff with only headers and no content lines normalizes to empty."""
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "empty.py",
                "diff": ("--- /dev/null\n+++ b/empty.py\n@@ -0,0 +1,0 @@"),
            }
        )
        # All headers stripped, empty diff creates empty file
        assert result.success is True
        assert (tmp_path / "empty.py").read_text() == ""

    @pytest.mark.asyncio
    async def test_triple_plus_in_content_not_stripped(self, tmp_path: Path) -> None:
        """+++ line appearing after content has started must NOT be stripped."""
        engine = _make_engine(tmp_path)
        result = await engine.execute(
            {
                "type": "create_file",
                "path": "hello.py",
                "diff": ("+first line\n+++ second line\n+third line"),
            }
        )
        assert result.success is True
        content = (tmp_path / "hello.py").read_text()
        # "+++ second line" is a content line (starts with +), NOT a header.
        # The parser strips the leading + to get "++ second line" as file content.
        assert content == "first line\n++ second line\nthird line"
