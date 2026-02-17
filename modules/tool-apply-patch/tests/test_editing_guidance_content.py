"""Content constraints for context/editing-guidance.md.

The editing-guidance context file should contain ONLY tool selection guidance
(when to use which tool). Format specifics (V4A markers, diff syntax, etc.)
belong in the tool description itself, not in the context file.
"""

from __future__ import annotations

from pathlib import Path

# The context file lives at the bundle root: .../amplifier-bundle-filesystem/context/
# Test file is at: .../amplifier-bundle-filesystem/modules/tool-apply-patch/tests/
_GUIDANCE_PATH = Path(__file__).resolve().parents[3] / "context" / "editing-guidance.md"


class TestEditingGuidanceContent:
    """editing-guidance.md must not contain V4A format details."""

    def _read_guidance(self) -> str:
        return _GUIDANCE_PATH.read_text(encoding="utf-8")

    def test_no_begin_patch_marker(self) -> None:
        content = self._read_guidance()
        assert "*** Begin Patch" not in content, (
            "Should not contain V4A Begin Patch marker"
        )

    def test_no_end_patch_marker(self) -> None:
        content = self._read_guidance()
        assert "*** End Patch" not in content, "Should not contain V4A End Patch marker"

    def test_no_at_at_diff_syntax(self) -> None:
        content = self._read_guidance()
        assert "@@" not in content, "Should not contain @@ diff syntax"

    def test_no_v4a_heading(self) -> None:
        content = self._read_guidance()
        assert "(V4A format)" not in content, "Should not reference V4A in heading"

    def test_has_tool_selection_decision_flow(self) -> None:
        content = self._read_guidance()
        assert "## Decision Flow" in content, (
            "Must contain tool selection decision flow"
        )

    def test_defers_format_to_tool_description(self) -> None:
        content = self._read_guidance()
        assert (
            "The tool description contains the complete format reference" in content
        ), "Must defer format details to tool description"

    def test_has_all_three_tools(self) -> None:
        content = self._read_guidance()
        assert "## apply_patch" in content
        assert "## edit_file" in content
        assert "## write_file" in content
