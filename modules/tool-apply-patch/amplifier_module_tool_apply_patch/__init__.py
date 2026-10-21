"""Apply Patch Tool Module for Amplifier.

Provides V4A diff-based file editing with two engine options:
- "function": Standard function tool, works with any provider
- "native": Integrates with OpenAI's built-in apply_patch tool type
"""

# Amplifier module metadata
__amplifier_module_type__ = "tool"

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from amplifier_core import ModuleCoordinator

__all__ = ["mount"]

logger = logging.getLogger(__name__)


async def mount(
    coordinator: "ModuleCoordinator", config: dict[str, Any] | None = None
) -> None:
    """Mount the apply_patch tool.

    Args:
        coordinator: Module coordinator for registering tools
        config: Module configuration. Keys:
            - engine: "native" | "function" (default: "native")
            - working_dir: explicit directory, otherwise session.working_dir
            - allowed_write_paths: list of allowed directories (default: [working_dir or "."])
            - denied_write_paths: list of denied directories (default: [])
            - additional_allowed_write_paths: extra grants (default: []), appended
              to the explicit allowlist (including []) or the mount-time default

    Present policy fields must be lists of strings. Policy lists and working_dir
    are resolved on a local copy; the caller's configuration is not mutated.
    Denies take precedence. Existing policy strings are not rebased.
    """
    config = dict(config or {})
    for field in (
        "allowed_write_paths",
        "denied_write_paths",
        "additional_allowed_write_paths",
    ):
        if field in config:
            paths = config[field]
            if not isinstance(paths, list) or any(
                not isinstance(path, str) for path in paths
            ):
                raise ValueError(
                    f"tool-apply-patch {field} must be a list of path strings"
                )
            config[field] = paths.copy()

    # Get session.working_dir capability if not explicitly configured
    if "working_dir" not in config:
        working_dir = coordinator.get_capability("session.working_dir")
        if working_dir:
            config["working_dir"] = working_dir
            logger.debug(f"Using session.working_dir: {working_dir}")

    additional = config.get("additional_allowed_write_paths", [])
    if additional:
        default_allowed = [config["working_dir"]] if config.get("working_dir") else ["."]
        paths = config.get("allowed_write_paths", default_allowed)
        config["allowed_write_paths"] = list(dict.fromkeys([*paths, *additional]))

    # Import here to avoid circular imports during module discovery
    from .tool import ApplyPatchTool

    tool = ApplyPatchTool(config, coordinator)
    await coordinator.mount("tools", tool, name=tool.name)

    # Register engine capability so providers can discover the active engine
    engine = config.get("engine", "native")
    coordinator.register_capability("apply_patch.engine", engine)

    logger.info(f"Mounted apply_patch tool (engine={engine})")
