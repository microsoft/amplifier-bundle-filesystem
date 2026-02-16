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
            - allowed_write_paths: list of allowed directories (default: [working_dir or "."])
            - denied_write_paths: list of denied directories (default: [])
    """
    config = config or {}

    # Get session.working_dir capability if not explicitly configured
    if "working_dir" not in config:
        working_dir = coordinator.get_capability("session.working_dir")
        if working_dir:
            config["working_dir"] = working_dir
            logger.debug(f"Using session.working_dir: {working_dir}")

    # Import here to avoid circular imports during module discovery
    from .tool import ApplyPatchTool

    tool = ApplyPatchTool(config, coordinator)
    await coordinator.mount("tools", tool, name=tool.name)

    # Register engine capability so providers can discover the active engine
    engine = config.get("engine", "native")
    coordinator.register_capability("apply_patch.engine", engine)

    logger.info(f"Mounted apply_patch tool (engine={engine})")
