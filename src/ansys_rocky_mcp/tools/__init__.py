"""Registration of every Rocky MCP tool."""

from __future__ import annotations

from typing import Any

from ansys_rocky_mcp.tools import (
    project_tools,
    results_tools,
    run_tools,
    session_tools,
    setup_tools,
)

__all__ = ["register_all"]


def register_all(mcp: Any) -> None:
    """Register every Rocky tool on the MCP server."""
    session_tools.register(mcp)
    project_tools.register(mcp)
    setup_tools.register(mcp)
    run_tools.register(mcp)
    results_tools.register(mcp)
