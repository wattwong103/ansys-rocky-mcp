"""Live Rocky check. Skips unless Ansys Rocky 2024 R2 or newer is installed."""

from __future__ import annotations

import os

import pytest
from fastmcp import Client

from ansys_rocky_mcp.backends.pyrocky_backend import (
    MINIMUM_ROCKY_VERSION,
    _detect_installations,
)
from ansys_rocky_mcp.config import RockyConfig
from ansys_rocky_mcp.server import RockyMCP


def _rocky_installed() -> bool:
    installations = _detect_installations()
    return any(version >= MINIMUM_ROCKY_VERSION for version in installations)


async def test_minimal_sequence_on_a_real_rocky_session(tmp_path) -> None:
    """The official minimal study runs headless through the MCP tools."""
    if not _rocky_installed():
        if os.environ.get("ROCKY_MCP_LIVE") == "1":
            pytest.fail(
                "ROCKY_MCP_LIVE=1 but no Ansys Rocky 2024 R2 or newer install was detected."
            )
        pytest.skip("No Ansys Rocky 2024 R2 or newer install was detected.")

    config = RockyConfig(backend="pyrocky", headless=True, python_session=False)
    server = RockyMCP(name="Ansys Rocky (live)", config=config)
    async with Client(server) as client:
        created = await client.call_tool("rocky_create_project")
        assert created.data["simulated"] is False

        particle = await client.call_tool("rocky_create_particle", {"sieve_size": 0.1})
        surface = await client.call_tool(
            "rocky_create_surface",
            {"kind": "circular", "radius": 0.3, "center": [0.0, 0.0, 0.0]},
        )
        await client.call_tool(
            "rocky_create_inlet",
            {
                "surface_id": surface.data["surface_id"],
                "particle_id": particle.data["particle_id"],
                "mass_flow_rate": 1000.0,
            },
        )
        await client.call_tool(
            "rocky_set_domain",
            {"use_boundary_limits": False, "max_values": [10.0, 1.0, 10.0]},
        )
        await client.call_tool("rocky_set_simulation_settings", {"duration": 2.0})
        await client.call_tool("rocky_save_project", {"path": str(tmp_path / "minimal.rocky")})
        await client.call_tool("rocky_run_simulation", {"non_blocking": False})
        curve = await client.call_tool("rocky_get_curve", {"name": "Particles Count"})

    assert curve.data["simulated"] is False
    assert curve.data["number_of_points"] >= 1
