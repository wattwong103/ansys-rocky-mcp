"""Tests for the server wiring: configuration, CLI, tools and the stdio transport."""

from __future__ import annotations

import os
import sys

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from ansys_rocky_mcp.backends import create_backend
from ansys_rocky_mcp.backends.base import RockyUnavailableError
from ansys_rocky_mcp.config import BACKEND_AUTO, RockyConfig
from ansys_rocky_mcp.registry import ObjectRegistry
from ansys_rocky_mcp.server import RockyMCP

EXPECTED_TOOLS = {
    "rocky_status",
    "rocky_connect",
    "rocky_disconnect",
    "rocky_create_project",
    "rocky_open_project",
    "rocky_save_project",
    "rocky_close_project",
    "rocky_describe_study",
    "rocky_import_wall",
    "rocky_create_surface",
    "rocky_create_particle",
    "rocky_create_inlet",
    "rocky_create_outlet",
    "rocky_set_domain",
    "rocky_set_simulation_settings",
    "rocky_set_motion",
    "rocky_run_simulation",
    "rocky_simulation_status",
    "rocky_wait_for_completion",
    "rocky_get_curve",
    "rocky_list_grid_functions",
    "rocky_get_grid_function_stats",
    "rocky_create_selection",
    "rocky_list_processes",
    "rocky_run_python",
}


def test_every_expected_tool_is_registered(mock_server: RockyMCP) -> None:
    """All tools of the public surface are registered."""
    import asyncio

    tools = asyncio.run(mock_server.list_tools())
    names = {tool.name for tool in tools}
    assert EXPECTED_TOOLS <= names


def test_config_reads_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variables configure the server."""
    monkeypatch.setenv("ROCKY_MCP_BACKEND", "mock")
    monkeypatch.setenv("ROCKY_MCP_PORT", "19999")
    monkeypatch.setenv("ROCKY_MCP_HEADLESS", "false")
    monkeypatch.setenv("ROCKY_MCP_VERSION", "261")

    config = RockyConfig.from_env()
    assert config.backend == "mock"
    assert config.rocky_port == 19999
    assert config.headless is False
    assert config.rocky_version == 261


def test_config_ignores_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Invalid environment values fall back to the defaults."""
    monkeypatch.setenv("ROCKY_MCP_PORT", "not-a-port")
    monkeypatch.setenv("ROCKY_MCP_HEADLESS", "maybe")

    config = RockyConfig.from_env()
    assert config.rocky_port == RockyConfig().rocky_port
    assert config.headless is True


def test_config_rejects_an_unknown_backend() -> None:
    """An unknown backend is rejected at construction time."""
    with pytest.raises(ValueError, match="Invalid backend"):
        RockyConfig(backend="quantum")


def test_merge_ignores_none_values() -> None:
    """Merging keeps the existing value when the override is ``None``."""
    config = RockyConfig(backend="mock", rocky_port=1234)
    merged = config.merge(backend=None, rocky_port=4321)
    assert merged.backend == "mock"
    assert merged.rocky_port == 4321


def test_auto_backend_fails_loudly_without_ansys() -> None:
    """The ``auto`` backend never silently substitutes simulated results."""
    config = RockyConfig(backend=BACKEND_AUTO)
    try:
        backend = create_backend(config)
    except RockyUnavailableError as exc:
        assert "mock" in str(exc)
        assert "simulated" in str(exc)
    else:
        # Ansys Rocky is installed on this machine, so the real backend is used.
        assert backend.mode == BACKEND_AUTO


@pytest.mark.asyncio
async def test_auto_backend_status_is_actionable() -> None:
    """Without Ansys installed, ``rocky_status`` explains what to do next."""
    server = RockyMCP(
        name="Ansys Rocky (auto)",
        config=RockyConfig(backend=BACKEND_AUTO, python_session=False),
    )
    async with Client(server) as client:
        status = await client.call_tool("rocky_status")
        if status.data["rocky_available"]:
            pytest.skip("Ansys Rocky is installed on this machine.")
        assert "--backend mock" in status.data["detail"]
        assert status.data["simulated"] is False


def test_cli_arguments_are_applied() -> None:
    """The CLI arguments reach the configuration."""
    server = RockyMCP(name="Ansys Rocky", config=RockyConfig(backend="mock"))
    import argparse

    parser = argparse.ArgumentParser()
    server._add_cli_arguments(parser)
    args = parser.parse_args(
        [
            "--backend",
            "mock",
            "--rocky-port",
            "19998",
            "--no-headless",
            "--no-python-session",
            "--mock-speed",
            "12",
        ]
    )
    server._configure_from_cli(args)

    assert server.config.backend == "mock"
    assert server.config.rocky_port == 19998
    assert server.config.headless is False
    assert server.config.python_session is False
    assert server.config.mock_speed == pytest.approx(12.0)


def test_registry_assigns_stable_identifiers() -> None:
    """The registry hands out unique, category-scoped identifiers."""
    registry = ObjectRegistry()
    first = registry.add("surface", object(), label="a")
    second = registry.add("surface", object(), label="b")
    third = registry.add("particle", object())

    assert first == "surface_1"
    assert second == "surface_2"
    assert third == "particle_1"
    assert len(registry) == 3
    assert registry.resolve(first) is not None

    with pytest.raises(KeyError, match="expected a 'particle'"):
        registry.resolve(first, "particle")


@pytest.mark.asyncio
async def test_stdio_transport_serves_a_real_mcp_session() -> None:
    """The server speaks MCP over stdio when launched as a subprocess."""
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "ansys_rocky_mcp", "--backend", "mock", "--no-python-session"],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    async with Client(transport) as client:
        tools = {tool.name for tool in await client.list_tools()}
        assert EXPECTED_TOOLS <= tools

        status = await client.call_tool("rocky_status")
        assert status.data["backend"] == "mock"
        assert status.data["simulated"] is True

        await client.call_tool("rocky_create_project")
        surface = await client.call_tool(
            "rocky_create_surface", {"kind": "circular", "radius": 0.3}
        )
        assert surface.data["surface_id"] == "surface_1"

        particle = await client.call_tool("rocky_create_particle", {"sieve_size": 0.05})
        await client.call_tool(
            "rocky_create_inlet",
            {
                "surface_id": surface.data["surface_id"],
                "particle_id": particle.data["particle_id"],
                "mass_flow_rate": 1000.0,
            },
        )
        await client.call_tool(
            "rocky_set_simulation_settings", {"duration": 1.0, "time_interval": 0.1}
        )
        await client.call_tool("rocky_run_simulation", {"non_blocking": False})

        curve = await client.call_tool("rocky_get_curve", {"name": "Particles Count"})
        assert curve.data["number_of_points"] > 1
