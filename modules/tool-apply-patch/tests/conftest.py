"""
Pytest configuration for module tests.

Behavioral tests use inheritance from amplifier-core base classes.
The amplifier-core pytest plugin provides fixtures automatically:
- module_path: Detected path to this module
- module_type: Detected type (provider, tool, hook, etc.)
- tool_module: Mounted module instance
"""
