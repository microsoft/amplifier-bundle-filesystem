"""Tests for shared path validation utilities.

Tests the allow/deny logic, symlink prevention, and relative path enforcement
that both engines use before any filesystem write.
"""

from __future__ import annotations

from pathlib import Path

from amplifier_module_tool_apply_patch.path_validation import (
    is_path_allowed,
    is_in_path_list,
)


class TestIsInPathList:
    """Tests for is_in_path_list helper."""

    def test_exact_match(self, tmp_path: Path) -> None:
        target = tmp_path / "project"
        target.mkdir()
        assert is_in_path_list(target, [str(target)]) is True

    def test_child_path_matches_parent(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        child = project / "src" / "main.py"
        child.parent.mkdir(parents=True)
        child.touch()
        assert is_in_path_list(child, [str(project)]) is True

    def test_sibling_path_does_not_match(self, tmp_path: Path) -> None:
        project_a = tmp_path / "project_a"
        project_b = tmp_path / "project_b"
        project_a.mkdir()
        project_b.mkdir()
        assert is_in_path_list(project_b / "file.py", [str(project_a)]) is False

    def test_parent_path_does_not_match_child_allow(self, tmp_path: Path) -> None:
        project = tmp_path / "project"
        project.mkdir()
        # Allowing only project/src should NOT allow project/ itself
        src = project / "src"
        src.mkdir()
        assert is_in_path_list(project / "file.py", [str(src)]) is False

    def test_empty_path_list_returns_false(self, tmp_path: Path) -> None:
        assert is_in_path_list(tmp_path / "anything", []) is False


class TestIsPathAllowed:
    """Tests for the main is_path_allowed function."""

    def test_allowed_path_returns_true(self, tmp_path: Path) -> None:
        allowed, error = is_path_allowed(
            tmp_path / "file.py",
            allowed_paths=[str(tmp_path)],
            denied_paths=[],
        )
        assert allowed is True
        assert error is None

    def test_denied_path_returns_false(self, tmp_path: Path) -> None:
        secrets = tmp_path / ".secrets"
        secrets.mkdir()
        allowed, error = is_path_allowed(
            secrets / "key.pem",
            allowed_paths=[str(tmp_path)],
            denied_paths=[str(secrets)],
        )
        assert allowed is False
        assert error is not None
        assert "denied" in error.lower()

    def test_deny_takes_priority_over_allow(self, tmp_path: Path) -> None:
        """Deny ALWAYS wins, even when path matches both allow and deny lists."""
        target = tmp_path / "project" / "secrets"
        target.mkdir(parents=True)
        allowed, error = is_path_allowed(
            target / "file.txt",
            allowed_paths=[str(tmp_path)],
            denied_paths=[str(target)],
        )
        assert allowed is False

    def test_path_not_in_allow_list_is_denied(self, tmp_path: Path) -> None:
        allowed, error = is_path_allowed(
            Path("/etc/passwd"),
            allowed_paths=[str(tmp_path)],
            denied_paths=[],
        )
        assert allowed is False
        assert error is not None

    def test_symlink_traversal_is_prevented(self, tmp_path: Path) -> None:
        """A symlink inside allowed dir pointing outside should be denied."""
        allowed_dir = tmp_path / "project"
        allowed_dir.mkdir()
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        secret = outside_dir / "secret.txt"
        secret.write_text("sensitive")

        # Create symlink inside allowed dir pointing to outside
        link = allowed_dir / "sneaky_link"
        link.symlink_to(secret)

        # The resolved path is outside allowed dirs
        allowed, error = is_path_allowed(
            link.resolve(),  # resolve follows the symlink
            allowed_paths=[str(allowed_dir)],
            denied_paths=[],
        )
        assert allowed is False

    def test_relative_path_resolved_against_working_dir(self, tmp_path: Path) -> None:
        """Relative paths should be resolved before checking."""
        project = tmp_path / "project"
        project.mkdir()
        # A relative path, when resolved from project/, should be allowed
        rel_path = project / "src" / "main.py"
        allowed, error = is_path_allowed(
            rel_path,
            allowed_paths=[str(project)],
            denied_paths=[],
        )
        assert allowed is True
