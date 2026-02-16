"""Structural and behavioral validation tests for apply_patch tool.

Inherits authoritative tests from amplifier-core, same pattern as
amplifier-module-tool-filesystem.
"""

from amplifier_core.validation.structural import ToolStructuralTests


class TestApplyPatchToolStructural(ToolStructuralTests):
    """Run standard tool structural tests for apply_patch.

    All tests from ToolStructuralTests run automatically.
    """
