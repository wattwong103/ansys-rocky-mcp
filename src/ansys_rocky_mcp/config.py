"""Configuration for the Ansys Rocky MCP server.

Settings are resolved with the following precedence (highest first):

1. Command line arguments (see :meth:`RockyMCP._configure_from_cli`).
2. Environment variables prefixed with ``ROCKY_MCP_``.
3. Built-in defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

__all__ = [
    "BACKEND_AUTO",
    "BACKEND_MOCK",
    "BACKEND_PYROCKY",
    "DEFAULT_ROCKY_PORT",
    "RockyConfig",
]

BACKEND_AUTO = "auto"
BACKEND_PYROCKY = "pyrocky"
BACKEND_MOCK = "mock"

VALID_BACKENDS = (BACKEND_AUTO, BACKEND_PYROCKY, BACKEND_MOCK)

#: Default PyRocky RPC port (matches ``ansys.rocky.core.client.PYROCKY_DEFAULT_PORT``).
DEFAULT_ROCKY_PORT = 18615

#: Default timeout, in seconds, for launching the Rocky application.
DEFAULT_LAUNCH_TIMEOUT = 300.0

#: Default timeout, in seconds, for attaching to an already running Rocky session.
DEFAULT_CONNECT_TIMEOUT = 30.0

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return default


def _env_int(name: str, default: int | None) -> int | None:
    """Read an integer environment variable."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class RockyConfig:
    """Runtime configuration for the Rocky MCP server.

    Parameters
    ----------
    backend : str
        Which backend drives the tools.

        * ``"auto"`` - use the real Rocky application when it is installed and
          licensed, otherwise raise an actionable error. Never silently
          substitutes simulated numbers for real ones.
        * ``"pyrocky"`` - always use a real Rocky application.
        * ``"mock"`` - use the built-in in-memory Rocky simulator. Every
          response is tagged with ``backend="mock"`` and ``simulated=true``.
    rocky_exe : str | None
        Explicit path to the Rocky executable. Auto-detected when ``None``.
    rocky_version : int | None
        Ansys version code of Rocky to launch (for example ``261`` for 2026 R1).
    rocky_port : int
        Port used for the PyRocky RPC server.
    headless : bool
        Whether to launch Rocky without a graphical user interface.
    python_session : bool
        Whether to start the persistent Python session used by
        :func:`ansys_rocky_mcp.tools.results_tools.rocky_run_python`.
    launch_timeout : float
        Seconds to wait for the Rocky application to become reachable.
    connect_timeout : float
        Seconds to wait when attaching to an already running Rocky session.
    mock_speed : float
        Simulated seconds per wall-clock second used by the mock backend.
    working_directory : str | None
        Working directory for the server and the persistent Python session.

    """

    backend: str = BACKEND_AUTO
    rocky_exe: str | None = None
    rocky_version: int | None = None
    rocky_port: int = DEFAULT_ROCKY_PORT
    headless: bool = True
    python_session: bool = True
    launch_timeout: float = DEFAULT_LAUNCH_TIMEOUT
    connect_timeout: float = DEFAULT_CONNECT_TIMEOUT
    mock_speed: float = 8.0
    working_directory: str | None = None

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.backend not in VALID_BACKENDS:
            raise ValueError(f"Invalid backend {self.backend!r}. Expected one of {VALID_BACKENDS}.")

    @classmethod
    def from_env(cls) -> RockyConfig:
        """Build a configuration from ``ROCKY_MCP_*`` environment variables."""
        return cls(
            backend=os.environ.get("ROCKY_MCP_BACKEND", BACKEND_AUTO).strip().lower()
            or BACKEND_AUTO,
            rocky_exe=os.environ.get("ROCKY_MCP_EXE") or None,
            rocky_version=_env_int("ROCKY_MCP_VERSION", None),
            rocky_port=_env_int("ROCKY_MCP_PORT", DEFAULT_ROCKY_PORT) or DEFAULT_ROCKY_PORT,
            headless=_env_bool("ROCKY_MCP_HEADLESS", True),
            python_session=_env_bool("ROCKY_MCP_PYTHON_SESSION", True),
            launch_timeout=_env_float("ROCKY_MCP_LAUNCH_TIMEOUT", DEFAULT_LAUNCH_TIMEOUT),
            connect_timeout=_env_float("ROCKY_MCP_CONNECT_TIMEOUT", DEFAULT_CONNECT_TIMEOUT),
            mock_speed=_env_float("ROCKY_MCP_MOCK_SPEED", 8.0),
            working_directory=os.environ.get("ROCKY_MCP_WORKDIR") or None,
        )

    def merge(self, **overrides: object) -> RockyConfig:
        """Return a copy of the configuration with the given fields replaced.

        ``None`` values are ignored so that unset CLI options never clobber
        values coming from the environment.
        """
        applied = {key: value for key, value in overrides.items() if value is not None}
        if "backend" in applied:
            applied["backend"] = str(applied["backend"]).strip().lower()
        return replace(self, **applied)

    def describe(self) -> dict[str, object]:
        """Return a JSON-serializable summary of the configuration."""
        return {
            "backend": self.backend,
            "rocky_exe": self.rocky_exe,
            "rocky_version": self.rocky_version,
            "rocky_port": self.rocky_port,
            "headless": self.headless,
            "python_session": self.python_session,
            "launch_timeout": self.launch_timeout,
            "connect_timeout": self.connect_timeout,
            "mock_speed": self.mock_speed,
            "working_directory": self.working_directory,
        }
