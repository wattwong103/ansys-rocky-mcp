"""Backend that drives a real Ansys Rocky application through PyRocky."""

from __future__ import annotations

from typing import Any

from ansys_rocky_mcp.backends.base import (
    BackendStatus,
    RockyBackend,
    RockyUnavailableError,
)
from ansys_rocky_mcp.config import RockyConfig

__all__ = ["PyRockyBackend"]

#: Oldest Ansys release that ships Ansys Rocky.
MINIMUM_ROCKY_VERSION = 242  # 2024 R2


def _detect_installations() -> dict[int, str]:
    """Return the locally installed Ansys products keyed by version code."""
    try:
        from ansys.tools.common.path.path import get_available_ansys_installations
    except Exception:  # pragma: no cover - depends on the local ansys-tools build
        return {}
    try:
        return dict(get_available_ansys_installations())
    except Exception:  # pragma: no cover - detection must never break the server
        return {}


class PyRockyBackend(RockyBackend):
    """Backend that launches or attaches to a real Ansys Rocky session."""

    mode = "pyrocky"
    simulated = False

    def __init__(self, config: RockyConfig) -> None:
        """Initialize the backend."""
        self._config = config
        self._client: Any = None
        self._app: Any = None

    # ------------------------------------------------------------------
    # Backend interface
    # ------------------------------------------------------------------
    def probe(self) -> BackendStatus:
        """Report which Rocky installations are available locally."""
        installations = _detect_installations()
        rocky_versions = sorted(
            version for version in installations if version >= MINIMUM_ROCKY_VERSION
        )
        available = bool(rocky_versions)
        detail = (
            f"Found Ansys installations suitable for Rocky: "
            f"{', '.join(str(version) for version in rocky_versions)}."
            if available
            else (
                "No Ansys installation with Rocky (2024 R2 or newer) was detected. "
                "Install Ansys Rocky and make sure the AWP_ROOT242 (or newer) "
                "environment variable is set."
            )
        )
        return BackendStatus(
            mode=self.mode,
            connected=self._client is not None,
            simulated=False,
            rocky_available=available,
            rocky_version=self._app.GetVersion() if self._app is not None else None,
            installed_versions=[str(version) for version in rocky_versions],
            detail=detail,
        )

    def connect(self, mode: str = "auto") -> Any:
        """Launch or attach to Rocky and return the application object.

        Parameters
        ----------
        mode : str, default: "auto"
            ``"launch"`` always starts a new Rocky application, ``"attach"``
            connects to an already running one, and ``"auto"`` attaches when a
            session is already listening and launches otherwise.

        """
        if self._app is not None:
            return self._app

        if mode == "attach":
            self._attach()
        elif mode == "launch":
            self._launch()
        elif mode == "auto":
            try:
                self._attach()
            except Exception:
                self._launch()
        else:
            raise ValueError(
                f"Unknown connection mode {mode!r}. Expected 'auto', 'launch' or 'attach'."
            )
        return self._app

    def close(self) -> None:
        """Close the Rocky client."""
        client = self._client
        self._client = None
        self._app = None
        if client is not None:
            try:
                client.close()
            except Exception:  # pragma: no cover - closing must never raise
                pass

    @property
    def app(self) -> Any:
        """Return the live Rocky application object."""
        if self._app is None:
            raise RockyUnavailableError(
                "No Ansys Rocky application is connected. Call 'rocky_connect' first."
            )
        return self._app

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _attach(self) -> None:
        """Connect to an already running Rocky session."""
        from ansys.rocky.core import client as rocky_client

        self._client = rocky_client.connect(port=self._config.rocky_port)
        self._app = self._client.api()

    def _launch(self) -> None:
        """Launch a new Rocky application."""
        from ansys.rocky.core import launcher as rocky_launcher

        kwargs: dict[str, Any] = {
            "headless": self._config.headless,
            "server_port": self._config.rocky_port,
            "connect_timeout": int(self._config.connect_timeout),
        }
        if self._config.rocky_exe:
            kwargs["rocky_exe"] = self._config.rocky_exe
        if self._config.rocky_version is not None:
            kwargs["rocky_version"] = self._config.rocky_version

        try:
            self._client = rocky_launcher.launch_rocky(**kwargs)
        except Exception as exc:
            raise RockyUnavailableError(
                "Could not launch Ansys Rocky. Check that Rocky 2024 R2 or newer is "
                f"installed and licensed, and that AWP_ROOT242 (or newer) is set. "
                f"Original error: {exc}"
            ) from exc
        self._app = self._client.api()
