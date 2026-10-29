"""Session lifecycle for the Ansys Rocky MCP server."""

from __future__ import annotations

from typing import Any

from ansys_rocky_mcp.backends import (
    BackendStatus,
    RockyBackend,
    RockyNotConnectedError,
    RockyUnavailableError,
    create_backend,
)
from ansys_rocky_mcp.config import RockyConfig
from ansys_rocky_mcp.registry import ObjectRegistry

__all__ = ["RockySession"]


class RockySession:
    """Own the connection to Rocky and the registry of live API objects.

    The connection is lazy: creating a session never launches Ansys Rocky.
    The first tool call that needs Rocky triggers the launch, which keeps
    server start-up fast and lets the client report the Rocky status first.
    """

    def __init__(self, config: RockyConfig) -> None:
        """Initialize an unconnected session."""
        self.config = config
        self.registry = ObjectRegistry()
        self._backend: RockyBackend | None = None
        self._app: Any = None

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    @property
    def backend(self) -> RockyBackend | None:
        """Return the active backend, or ``None`` when not connected."""
        return self._backend

    @property
    def simulated(self) -> bool:
        """Return whether the session produces simulated results."""
        return bool(self._backend is not None and self._backend.simulated)

    @property
    def connected(self) -> bool:
        """Return whether a Rocky application is available."""
        return self._app is not None

    @property
    def app(self) -> Any:
        """Return the live Rocky application object.

        Raises
        ------
        RockyNotConnectedError
            When no Rocky application has been connected yet.

        """
        if self._app is None:
            raise RockyNotConnectedError(
                "No Ansys Rocky application is connected yet. Call 'rocky_connect' "
                "first, or run the server with '--backend mock' to use the built-in "
                "simulator."
            )
        return self._app

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def connect(self, mode: str = "auto") -> BackendStatus:
        """Connect to Rocky and return the resulting status."""
        if self._backend is None:
            self._backend = create_backend(self.config)
        self._app = self._backend.connect(mode)
        self.registry.clear()
        return self.status()

    def ensure_app(self) -> Any:
        """Return the Rocky application, connecting on first use."""
        if self._app is None:
            self.connect("auto")
        return self._app

    def close(self) -> None:
        """Disconnect from Rocky and clear the registry."""
        if self._backend is not None:
            self._backend.close()
        self._backend = None
        self._app = None
        self.registry.clear()

    def status(self) -> BackendStatus:
        """Return the current session status."""
        if self._backend is None:
            try:
                backend = create_backend(self.config)
            except RockyUnavailableError as exc:
                return BackendStatus(
                    mode=self.config.backend,
                    connected=False,
                    simulated=False,
                    rocky_available=False,
                    detail=str(exc),
                )
            status = backend.probe()
            status.mode = self.config.backend
            return status

        status = self._backend.probe()
        status.connected = self._app is not None
        status.simulated = self._backend.simulated
        if self._app is not None:
            try:
                status.rocky_version = self._app.GetVersion()
            except Exception:  # pragma: no cover - version lookup is best effort
                pass
        status.project_path = self._safe_project_path()
        status.simulating, status.progress = self._safe_simulation_state()
        return status

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    def get_project(self) -> Any:
        """Return the open project, raising a helpful error when there is none."""
        app = self.ensure_app()
        project = app.GetProject()
        if project is None:
            raise RuntimeError(
                "There is no open Rocky project. Call 'rocky_create_project' or "
                "'rocky_open_project' first."
            )
        return project

    def get_study(self) -> Any:
        """Return the active study of the open project."""
        return self.get_project().GetStudy()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _safe_project_path(self) -> str | None:
        """Return the project file path without raising."""
        if self._app is None:
            return None
        try:
            project = self._app.GetProject()
        except Exception:  # pragma: no cover - status must never raise
            return None
        if project is None:
            return None
        try:
            filename = project.GetProjectFilename()
        except Exception:  # pragma: no cover
            return None
        return filename or None

    def _safe_simulation_state(self) -> tuple[bool, float | None]:
        """Return the ``(simulating, progress)`` pair without raising."""
        if self._app is None:
            return False, None
        try:
            study = self._app.GetProject().GetStudy()
            return bool(study.IsSimulating()), study.GetProgress()
        except Exception:  # pragma: no cover
            return False, None
