"""Backends used to drive Ansys Rocky."""

from __future__ import annotations

from ansys_rocky_mcp.backends.base import (
    BackendStatus,
    RockyBackend,
    RockyBackendError,
    RockyNotConnectedError,
    RockyUnavailableError,
)
from ansys_rocky_mcp.backends.mock_backend import MockBackend, MockRockyApplication
from ansys_rocky_mcp.backends.pyrocky_backend import PyRockyBackend
from ansys_rocky_mcp.config import BACKEND_AUTO, BACKEND_MOCK, BACKEND_PYROCKY, RockyConfig

__all__ = [
    "BACKEND_AUTO",
    "BACKEND_MOCK",
    "BACKEND_PYROCKY",
    "BackendStatus",
    "MockBackend",
    "MockRockyApplication",
    "PyRockyBackend",
    "RockyBackend",
    "RockyBackendError",
    "RockyNotConnectedError",
    "RockyUnavailableError",
    "create_backend",
]


def create_backend(config: RockyConfig) -> RockyBackend:
    """Create the backend described by ``config``.

    Parameters
    ----------
    config : RockyConfig
        The server configuration.

    Returns
    -------
    RockyBackend
        The backend instance.

    Raises
    ------
    RockyUnavailableError
        When the ``auto`` backend finds no local Ansys Rocky installation.

    """
    if config.backend == BACKEND_MOCK:
        return MockBackend(speed=config.mock_speed)

    if config.backend == BACKEND_PYROCKY:
        return PyRockyBackend(config)

    # BACKEND_AUTO: use a real Rocky when one is installed, otherwise fail loudly.
    candidate = PyRockyBackend(config)
    status = candidate.probe()
    if status.rocky_available:
        return candidate
    raise RockyUnavailableError(
        "Ansys Rocky was not detected on this machine. Install Ansys Rocky "
        "(2024 R2 or newer) and set AWP_ROOT242 (or newer), then start the "
        "server again. To explore the tool interface without Ansys installed, "
        "run the server with '--backend mock'; every response is then tagged "
        "with backend='mock' and simulated=true."
    )
