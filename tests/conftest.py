"""Shared pytest fixtures for the Ansys Rocky MCP test suite."""

from __future__ import annotations

import pytest
from fastmcp import Client

from ansys_rocky_mcp.config import RockyConfig
from ansys_rocky_mcp.server import RockyMCP


@pytest.fixture
def mock_config() -> RockyConfig:
    """Return a configuration that drives the in-memory Rocky simulator."""
    return RockyConfig(
        backend="mock",
        mock_speed=400.0,
        python_session=False,
    )


@pytest.fixture
def mock_server(mock_config: RockyConfig) -> RockyMCP:
    """Return a Rocky MCP server backed by the mock simulator."""
    return RockyMCP(name="Ansys Rocky (test)", config=mock_config)


@pytest.fixture
async def client(mock_server: RockyMCP):
    """Return an in-memory MCP client connected to the mock server."""
    async with Client(mock_server) as mcp_client:
        yield mcp_client


@pytest.fixture
async def study(client: Client):
    """Return a client with a ready-to-run study.

    The study contains a circular inlet surface, a spherical particle, a
    particle inlet with a mass flow of 1000 t/h, an outlet and a two second
    simulation window.
    """
    await client.call_tool("rocky_create_project")

    surface = await client.call_tool(
        "rocky_create_surface",
        {"kind": "circular", "radius": 0.3, "center": [0.0, 0.0, 0.0]},
    )
    outlet_surface = await client.call_tool(
        "rocky_create_surface",
        {"kind": "rectangular", "length": 3.0, "width": 3.0, "center": [5.0, 0.0, 0.0]},
    )
    particle = await client.call_tool(
        "rocky_create_particle", {"shape": "sphere", "sieve_size": 0.05}
    )

    await client.call_tool(
        "rocky_create_inlet",
        {
            "surface_id": surface.data["surface_id"],
            "particle_id": particle.data["particle_id"],
            "mass_flow_rate": 1000.0,
        },
    )
    await client.call_tool("rocky_create_outlet", {"surface_id": outlet_surface.data["surface_id"]})
    await client.call_tool(
        "rocky_set_domain",
        {"min_values": [-5.0, -1.5, -1.5], "max_values": [5.0, 1.5, 1.5]},
    )
    await client.call_tool("rocky_set_simulation_settings", {"duration": 2.0, "time_interval": 0.1})
    return client
