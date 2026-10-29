"""Shared helpers for the Rocky MCP tools."""

from __future__ import annotations

from typing import Any

from fastmcp import Context
from fastmcp.exceptions import ToolError

from ansys_rocky_mcp.session import RockySession

__all__ = [
    "envelope",
    "get_session",
    "require_project",
    "require_study",
    "to_float_list",
]


def get_session(ctx: Context) -> RockySession:
    """Return the Rocky session carried by the MCP lifespan context.

    Raises
    ------
    ToolError
        When the session has not been initialized.

    """
    lifespan = getattr(ctx.request_context, "lifespan_context", None)
    session = getattr(lifespan, "session", None)
    if session is None:
        raise ToolError(
            "The Rocky session is not initialized. Restart the MCP server; the "
            "session is created during server startup."
        )
    return session


def require_project(session: RockySession) -> Any:
    """Return the open Rocky project.

    Raises
    ------
    ToolError
        When no project is open.

    """
    try:
        return session.get_project()
    except RuntimeError as exc:
        raise ToolError(str(exc)) from exc


def require_study(session: RockySession) -> Any:
    """Return the active Rocky study.

    Raises
    ------
    ToolError
        When no project is open.

    """
    try:
        return session.get_study()
    except RuntimeError as exc:
        raise ToolError(str(exc)) from exc


def envelope(session: RockySession, **payload: Any) -> dict[str, Any]:
    """Wrap a tool payload with the backend provenance tags.

    Every response carries ``backend`` and ``simulated`` so that an AI client
    can never present simulated numbers as real Rocky results.
    """
    return {
        "backend": session.backend.mode if session.backend else session.config.backend,
        "simulated": session.simulated,
        **payload,
    }


def to_float_list(values: Any) -> list[float]:
    """Convert an array-like value into a list of floats."""
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        return [float(value) for value in values]
    except TypeError:
        return [float(values)]
