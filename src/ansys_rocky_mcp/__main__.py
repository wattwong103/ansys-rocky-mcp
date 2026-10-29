"""Command line entry point of the Ansys Rocky MCP server."""

from __future__ import annotations

from ansys_rocky_mcp.server import INSTRUCTIONS, RockyMCP

__all__ = ["main"]


def main() -> None:
    """Run the Rocky MCP server from the command line."""
    app = RockyMCP(name="Ansys Rocky", instructions=INSTRUCTIONS)
    app.run_cli()


if __name__ == "__main__":
    main()
