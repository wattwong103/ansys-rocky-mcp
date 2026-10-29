"""Backend abstraction used to drive Ansys Rocky."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "BackendStatus",
    "RockyBackend",
    "RockyBackendError",
    "RockyNotConnectedError",
    "RockyUnavailableError",
]


class RockyBackendError(RuntimeError):
    """Base class for backend failures."""


class RockyUnavailableError(RockyBackendError):
    """Raised when the requested backend cannot be used on this machine."""


class RockyNotConnectedError(RockyBackendError):
    """Raised when an operation needs a live Rocky session but none exists."""


@dataclass
class BackendStatus:
    """Snapshot of the backend state reported by ``rocky_status``."""

    mode: str
    connected: bool = False
    simulated: bool = False
    rocky_available: bool = False
    rocky_version: str | None = None
    installed_versions: list[str] = field(default_factory=list)
    detail: str = ""
    project_path: str | None = None
    simulating: bool = False
    progress: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the status."""
        return {
            "backend": self.mode,
            "connected": self.connected,
            "simulated": self.simulated,
            "rocky_available": self.rocky_available,
            "rocky_version": self.rocky_version,
            "installed_versions": list(self.installed_versions),
            "detail": self.detail,
            "project_path": self.project_path,
            "simulating": self.simulating,
            "progress": self.progress,
        }


class RockyBackend(ABC):
    """Interface implemented by the real and mock Rocky backends.

    A backend exposes the Rocky *application* object - the same object that
    ``pyrocky`` returns from ``RockyClient.api()``. Its methods are
    ``CreateProject()``, ``GetProject()``, ``GetVersion()`` and friends. Tool
    code is written against that single interface, so it behaves identically
    whether a real Rocky application or the mock simulator is behind it.
    """

    #: Backend identifier reported by ``rocky_status``.
    mode: str = "unknown"

    #: Whether the backend produces simulated rather than physical results.
    simulated: bool = False

    @abstractmethod
    def probe(self) -> BackendStatus:
        """Report what this backend can do without starting a simulation."""

    @abstractmethod
    def connect(self, mode: str = "auto") -> Any:
        """Start (or attach to) Rocky and return the application object.

        Parameters
        ----------
        mode : str, default: "auto"
            ``"auto"``, ``"launch"`` or ``"attach"``.

        """

    @abstractmethod
    def close(self) -> None:
        """Release any resources held by the backend."""

    @property
    def app(self) -> Any:
        """Return the live Rocky application object."""
        raise RockyNotConnectedError(
            "No Rocky application is connected. Call 'rocky_connect' first."
        )
