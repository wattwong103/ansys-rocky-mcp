"""Ansys Rocky MCP server.

A Model Context Protocol server that exposes Ansys Rocky, the discrete
element method (DEM) solver, to AI clients such as Claude, MiniMax Code,
Codex and Kimi.

The server follows the extension pattern of the official PyAnsys MCP servers
and is built on :mod:`ansys.common.mcp`.
"""

from __future__ import annotations

__version__ = "0.1.0"

from ansys_rocky_mcp.server import INSTRUCTIONS, RockyMCP

__all__ = ["INSTRUCTIONS", "RockyMCP", "__version__"]
