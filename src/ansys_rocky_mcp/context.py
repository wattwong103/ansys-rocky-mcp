"""Application context for the Ansys Rocky MCP server."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ansys.common.mcp.context import PyAnsysBaseAppContext

from ansys_rocky_mcp.config import RockyConfig
from ansys_rocky_mcp.session import RockySession

__all__ = ["RockyContext"]


@dataclass
class RockyContext(PyAnsysBaseAppContext):
    """Context holding the Rocky session for the lifetime of the server.

    The base class ``product_instance`` field carries the
    :class:`~ansys_rocky_mcp.session.RockySession`, matching the convention
    used by the other PyAnsys MCP servers. The typed ``session`` alias is
    provided for convenience.
    """

    session: RockySession | None = None
    config: RockyConfig | None = None

    @property
    def rocky_session(self) -> RockySession | None:
        """Return the Rocky session, or ``None`` when it is not initialized."""
        return self.session

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of the context."""
        return {
            "backend": self.config.backend if self.config else None,
            "has_session": self.session is not None,
            "has_python_session": self.python_session is not None,
            "command_history_length": len(self.command_history),
        }
