"""PyRocky backend tests. No Rocky process is started."""

from __future__ import annotations

import pytest

from ansys_rocky_mcp.backends.base import RockyUnavailableError
from ansys_rocky_mcp.backends.pyrocky_backend import PyRockyBackend
from ansys_rocky_mcp.config import RockyConfig


class _Application:
    """Stand-in whose call count shows whether ``api()`` was used."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> str:
        self.calls += 1
        return "called"


class _Client:
    """Fake ``RockyClient`` whose ``api`` is a property."""

    def __init__(self, app: _Application) -> None:
        self._app = app
        self.reads = 0

    @property
    def api(self) -> _Application:
        self.reads += 1
        return self._app


def test_attach_stores_the_api_property(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attach reads ``client.api`` once and does not call it."""
    app = _Application()
    client = _Client(app)

    def connect(port: int) -> _Client:
        assert port == RockyConfig().rocky_port
        return client

    monkeypatch.setattr("ansys.rocky.core.client.connect", connect)
    backend = PyRockyBackend(RockyConfig(backend="pyrocky"))

    assert backend.connect("attach") is app
    assert backend.app is app
    assert client.reads == 1
    assert app.calls == 0


def test_launch_passes_launch_timeout_as_connect_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """A default launch waits ``launch_timeout`` seconds, not the 30s attach timeout."""
    app = _Application()
    client = _Client(app)
    captured: dict[str, object] = {}

    def launch_rocky(**kwargs: object) -> _Client:
        captured.update(kwargs)
        return client

    monkeypatch.setattr("ansys.rocky.core.launcher.launch_rocky", launch_rocky)
    backend = PyRockyBackend(RockyConfig(backend="pyrocky"))

    assert backend.connect("launch") is app
    assert captured["connect_timeout"] == 300
    assert captured["connect_timeout"] != 30
    assert client.reads == 1
    assert app.calls == 0


def test_launch_forwards_a_configured_launch_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """``ROCKY_MCP_LAUNCH_TIMEOUT`` is the cold-start wait passed to ``launch_rocky``."""
    captured: dict[str, object] = {}

    def launch_rocky(**kwargs: object) -> _Client:
        captured.update(kwargs)
        return _Client(_Application())

    monkeypatch.setattr("ansys.rocky.core.launcher.launch_rocky", launch_rocky)
    config = RockyConfig(backend="pyrocky", launch_timeout=45)
    PyRockyBackend(config).connect("launch")

    assert captured["connect_timeout"] == 45


def test_failed_launch_raises_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed launch still reports the install and license hint."""

    def launch_rocky(**kwargs: object) -> _Client:
        raise RuntimeError("boom")

    monkeypatch.setattr("ansys.rocky.core.launcher.launch_rocky", launch_rocky)
    backend = PyRockyBackend(RockyConfig(backend="pyrocky"))

    with pytest.raises(RockyUnavailableError, match="licensed"):
        backend.connect("launch")
