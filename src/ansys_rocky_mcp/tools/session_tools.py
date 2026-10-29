"""Session and connection tools for the Rocky MCP server."""

from __future__ import annotations

from typing import Any

from fastmcp import Context
from fastmcp.exceptions import ToolError

from ansys_rocky_mcp.backends.base import RockyUnavailableError
from ansys_rocky_mcp.tools._common import envelope, get_session

__all__ = ["register"]


async def rocky_status(ctx: Context) -> dict[str, Any]:
    """Report the status of the Rocky MCP session.

    Returns the active backend, whether a real Ansys Rocky application is
    installed and reachable, the detected Rocky versions, the open project
    and the current simulation state.

    Use this first: it is the only tool that never launches Rocky.
    """
    session = get_session(ctx)
    status = session.status()
    payload = status.to_dict()
    payload["config"] = session.config.describe()
    payload["registered_objects"] = session.registry.list()
    return payload


async def rocky_connect(
    ctx: Context,
    mode: str = "auto",
    version: int | None = None,
    exe: str | None = None,
    port: int | None = None,
) -> dict[str, Any]:
    """Connect to Ansys Rocky.

    Parameters
    ----------
    mode : str, default: "auto"
        ``"auto"`` attaches to a running Rocky session when one is listening
        and launches a new one otherwise, ``"launch"`` always starts a fresh
        Rocky application, and ``"attach"`` only connects to an existing one.
    version : int | None, default: None
        Ansys version code to launch, for example ``261`` for 2026 R1.
    exe : str | None, default: None
        Explicit path to the Rocky executable.
    port : int | None, default: None
        Port for the PyRocky RPC server. Defaults to the server configuration.

    """
    session = get_session(ctx)
    if version is not None:
        session.config = session.config.merge(rocky_version=version)
    if exe:
        session.config = session.config.merge(rocky_exe=exe)
    if port is not None:
        session.config = session.config.merge(rocky_port=port)

    try:
        status = session.connect(mode)
    except RockyUnavailableError as exc:
        raise ToolError(str(exc)) from exc
    return envelope(session, **status.to_dict())


async def rocky_disconnect(ctx: Context) -> dict[str, Any]:
    """Disconnect from Ansys Rocky and release the session."""
    session = get_session(ctx)
    was_connected = session.connected
    session.close()
    return envelope(
        session,
        disconnected=was_connected,
        message="Rocky session closed." if was_connected else "No Rocky session was open.",
    )


def register(mcp: Any) -> None:
    """Register the session tools on the MCP server."""
    mcp.tool()(rocky_status)
    mcp.tool()(rocky_connect)
    mcp.tool()(rocky_disconnect)
