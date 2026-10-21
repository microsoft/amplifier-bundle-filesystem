"""Tool and module-mount unit tests with a fake coordinator boundary.

Real tools and engines are used; these tests do not qualify compiled coordination.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

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


@pytest.fixture(params=["function", "native"])
def mount_engine(request) -> str:
    return request.param


@pytest.fixture
def policy_dirs(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    """Use private, separate process/session/grant/outside directories."""
    root = tmp_path / "mount-policy"
    root.mkdir(mode=0o700)
    paths = {
        name: root / name
        for name in ("process", "session", "second-session", "grant", "outside")
    }
    for path in paths.values():
        path.mkdir(mode=0o700)
    paths["protected"] = paths["grant"] / "protected"
    paths["protected"].mkdir(mode=0o700)
    monkeypatch.chdir(paths["process"])
    assert Path.cwd() != paths["session"]
    return paths


def _mount_coordinator(working_dir: Any) -> MagicMock:
    coordinator = MagicMock()
    coordinator.get_capability.return_value = working_dir
    coordinator.mount = AsyncMock(return_value=None)
    coordinator.register_capability.return_value = None
    coordinator.hooks.emit = AsyncMock(return_value=None)
    return coordinator


async def _mounted_tool(config, session_dir) -> tuple[ApplyPatchTool, MagicMock]:
    coordinator = _mount_coordinator(session_dir)
    assert await mount(coordinator, config) is None
    coordinator.mount.assert_awaited_once()
    args = coordinator.mount.await_args
    assert args.args[0] == "tools"
    tool = args.args[1]
    assert isinstance(tool, ApplyPatchTool)
    assert args.kwargs == {"name": "apply_patch"}
    coordinator.register_capability.assert_called_once_with(
        "apply_patch.engine", tool.config.get("engine", "native")
    )
    return tool, coordinator


@pytest.mark.asyncio
@pytest.mark.parametrize("allow_scope", ["omitted", "empty", "relative"])
async def test_mount_resolves_additional_grants_after_session_directory(
    policy_dirs, mount_engine, allow_scope
):
    session = str(policy_dirs["session"])
    grant = str(policy_dirs["grant"])
    config = {
        "engine": mount_engine,
        "additional_allowed_write_paths": [grant, grant],
        "denied_write_paths": [str(policy_dirs["protected"]), "relative-deny"],
        "options": {"labels": ["unchanged"]},
    }
    if allow_scope == "empty":
        config["allowed_write_paths"] = []
    elif allow_scope == "relative":
        config["allowed_write_paths"] = [".", "relative-output", "."]
    original = deepcopy(config)

    tool, coordinator = await _mounted_tool(config, session)

    coordinator.get_capability.assert_called_once_with("session.working_dir")
    expected_base = {
        "omitted": [session],
        "empty": [],
        "relative": [".", "relative-output"],
    }[allow_scope]
    assert tool._engine.allowed_write_paths == [*expected_base, grant]
    assert tool._engine.working_dir == session
    assert isinstance(
        tool._engine, FunctionEngine if mount_engine == "function" else NativeEngine
    )
    assert tool._engine.denied_write_paths == config["denied_write_paths"]
    assert tool.config is not config
    assert tool.config["options"] is config["options"]  # Only policy lists need copies.
    for field in (
        "allowed_write_paths",
        "denied_write_paths",
        "additional_allowed_write_paths",
    ):
        if field in config:
            assert tool.config[field] is not config[field]
    tool._engine.allowed_write_paths.append("mounted-only")
    tool._engine.denied_write_paths.append("mounted-deny")
    tool.config["additional_allowed_write_paths"].append("mounted-extra")
    assert config == original
    assert "working_dir" not in config


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_additional", [False, True])
@pytest.mark.parametrize("deny_only", [False, True])
async def test_mount_without_additional_grants_leaves_engine_default_implicit(
    policy_dirs, mount_engine, empty_additional, deny_only
):
    config = {"engine": mount_engine}
    if empty_additional:
        config["additional_allowed_write_paths"] = []
    if deny_only:
        config["denied_write_paths"] = [str(policy_dirs["protected"])]
    original = deepcopy(config)
    session = str(policy_dirs["session"])

    tool, _ = await _mounted_tool(config, session)

    assert "allowed_write_paths" not in tool.config
    assert tool._engine.allowed_write_paths == [session]
    assert tool._engine.denied_write_paths == config.get("denied_write_paths", [])
    if deny_only:
        assert tool._engine.denied_write_paths is not config["denied_write_paths"]
    if empty_additional:
        assert tool.config["additional_allowed_write_paths"] is not (
            config["additional_allowed_write_paths"]
        )
    else:
        assert "additional_allowed_write_paths" not in tool.config
    assert config == original


@pytest.mark.asyncio
@pytest.mark.parametrize("capability_present", [False, True])
@pytest.mark.parametrize("configured_dir", ["omitted", "null", "empty", "string", "path"])
async def test_mount_preserves_working_directory_key_presence_precedence(
    policy_dirs, mount_engine, capability_present, configured_dir
):
    session = str(policy_dirs["session"]) if capability_present else None
    grant = str(policy_dirs["grant"])
    config = {"engine": mount_engine, "additional_allowed_write_paths": [grant]}
    if configured_dir != "omitted":
        config["working_dir"] = {
            "null": None,
            "empty": "",
            "string": str(policy_dirs["second-session"]),
            "path": policy_dirs["second-session"],
        }[configured_dir]
    original = deepcopy(config)

    tool, coordinator = await _mounted_tool(config, session)

    if configured_dir == "omitted":
        coordinator.get_capability.assert_called_once_with("session.working_dir")
        expected_dir = session
        assert ("working_dir" in tool.config) == capability_present
    else:
        coordinator.get_capability.assert_not_called()
        expected_dir = config["working_dir"]
        assert "working_dir" in tool.config
    assert tool._engine.working_dir == expected_dir
    assert tool._engine.allowed_write_paths == [
        expected_dir if expected_dir else ".",
        grant,
    ]
    assert config == original


@pytest.mark.asyncio
async def test_repeated_mount_of_same_input_resolves_each_session_freshly(
    policy_dirs, mount_engine
):
    grant = str(policy_dirs["grant"])
    config = {
        "engine": mount_engine,
        "additional_allowed_write_paths": [grant],
        "denied_write_paths": [str(policy_dirs["protected"])],
    }
    original = deepcopy(config)
    first_dir = str(policy_dirs["session"])
    second_dir = str(policy_dirs["second-session"])

    first, _ = await _mounted_tool(config, first_dir)
    second, _ = await _mounted_tool(config, second_dir)

    assert first._engine.allowed_write_paths == [first_dir, grant]
    assert second._engine.allowed_write_paths == [second_dir, grant]
    assert first.config is not second.config
    for field in (
        "allowed_write_paths",
        "denied_write_paths",
        "additional_allowed_write_paths",
    ):
        assert first.config[field] is not second.config[field]
    first._engine.allowed_write_paths.append("first-only")
    first._engine.denied_write_paths.append("first-deny")
    first.config["additional_allowed_write_paths"].append("first-extra")
    assert second._engine.allowed_write_paths == [second_dir, grant]
    assert second._engine.denied_write_paths == config["denied_write_paths"]
    assert second.config["additional_allowed_write_paths"] == [grant]
    assert config == original
    assert "working_dir" not in config
    assert "allowed_write_paths" not in config


@pytest.mark.asyncio
async def test_mount_accepts_missing_config_without_freezing_session_default(policy_dirs):
    tool, _ = await _mounted_tool(None, str(policy_dirs["session"]))
    assert tool._engine.allowed_write_paths == [str(policy_dirs["session"])]
    assert "allowed_write_paths" not in tool.config
    assert "additional_allowed_write_paths" not in tool.config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field",
    ["allowed_write_paths", "denied_write_paths", "additional_allowed_write_paths"],
)
@pytest.mark.parametrize(
    "malformed_kind",
    ["null", "absolute-string", "mapping", "null-item", "integer-item", "nested-list"],
)
async def test_mount_rejects_malformed_policy_before_mount_or_registration(
    policy_dirs, mount_engine, field, malformed_kind
):
    private_value = str(policy_dirs["outside"] / "private-value")
    malformed = {
        "null": None,
        "absolute-string": private_value,
        "mapping": {"path": private_value},
        "null-item": [private_value, None],
        "integer-item": [private_value, 1],
        "nested-list": [[private_value]],
    }[malformed_kind]
    config = {
        "engine": mount_engine,
        "allowed_write_paths": [],
        "denied_write_paths": [str(policy_dirs["protected"])],
        "additional_allowed_write_paths": [str(policy_dirs["grant"])],
        field: malformed,
    }
    original = deepcopy(config)
    coordinator = _mount_coordinator(str(policy_dirs["session"]))

    with pytest.raises(ValueError) as exc:
        await mount(coordinator, config)

    assert str(exc.value) == f"tool-apply-patch {field} must be a list of path strings"
    assert private_value not in str(exc.value)
    coordinator.mount.assert_not_called()
    coordinator.register_capability.assert_not_called()
    coordinator.get_capability.assert_not_called()
    assert config == original


def _create_input(engine: str, path: str) -> dict[str, str]:
    if engine == "native":
        return {"type": "create_file", "path": path, "diff": "+created"}
    marker = "***"
    return {
        "patch": "\n".join(
            [
                f"{marker} Begin Patch",
                f"{marker} Add File: {path}",
                "+created",
                f"{marker} End Patch",
            ]
        )
    }


@pytest_asyncio.fixture
async def mounted_policy_tool(policy_dirs, mount_engine) -> ApplyPatchTool:
    """Mount the real consumer; only coordinator registration/hooks are mocked."""
    tool, _ = await _mounted_tool(
        {
            "engine": mount_engine,
            "additional_allowed_write_paths": [str(policy_dirs["grant"])],
            "denied_write_paths": [str(policy_dirs["protected"])],
        },
        str(policy_dirs["session"]),
    )
    return tool


@pytest.mark.asyncio
async def test_mounted_policy_writes_session_and_additional_grant(
    policy_dirs, mount_engine, mounted_policy_tool
):
    session_file = policy_dirs["session"] / "created.txt"
    grant_file = policy_dirs["grant"] / "created.txt"
    session_result = await mounted_policy_tool.execute(
        _create_input(mount_engine, "created.txt")
    )
    grant_result = await mounted_policy_tool.execute(
        _create_input(mount_engine, str(grant_file))
    )
    assert session_result.success is True
    assert grant_result.success is True
    assert session_file.read_text() == "created"
    assert grant_file.read_text() == "created"
    assert not (policy_dirs["process"] / "created.txt").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("target_dir", ["process", "outside", "protected"])
async def test_mounted_policy_rejects_outside_and_denied_grant_subtree(
    policy_dirs, mount_engine, mounted_policy_tool, target_dir
):
    target = policy_dirs[target_dir] / "rejected.txt"
    result = await mounted_policy_tool.execute(_create_input(mount_engine, str(target)))
    assert result.success is False
    assert not target.exists()
    if target_dir == "protected":
        assert "within denied directories" in result.error["message"]


@pytest.mark.asyncio
async def test_mounted_explicit_empty_allowlist_does_not_gain_session_scope(
    policy_dirs, mount_engine
):
    tool, _ = await _mounted_tool(
        {
            "engine": mount_engine,
            "allowed_write_paths": [],
            "additional_allowed_write_paths": [str(policy_dirs["grant"])],
        },
        str(policy_dirs["session"]),
    )
    rejected = await tool.execute(_create_input(mount_engine, "rejected.txt"))
    grant_file = policy_dirs["grant"] / "created.txt"
    granted = await tool.execute(_create_input(mount_engine, str(grant_file)))
    assert rejected.success is False
    assert not (policy_dirs["session"] / "rejected.txt").exists()
    assert granted.success is True
    assert grant_file.read_text() == "created"


@pytest.mark.asyncio
async def test_mounted_policy_rejects_symlink_escape(
    policy_dirs, mount_engine, mounted_policy_tool
):
    link = policy_dirs["session"] / "escape"
    try:
        link.symlink_to(policy_dirs["outside"], target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink escape test requires platform symlink support")
    result = await mounted_policy_tool.execute(
        _create_input(mount_engine, "escape/rejected.txt")
    )
    assert result.success is False
    assert not (policy_dirs["outside"] / "rejected.txt").exists()
