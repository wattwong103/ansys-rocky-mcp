"""Tests for the persistent Python session escape hatch."""

from __future__ import annotations

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from ansys_rocky_mcp.config import RockyConfig
from ansys_rocky_mcp.server import RockyMCP


@pytest.fixture
async def python_client():
    """Return a client whose server runs a persistent Python session."""
    config = RockyConfig(backend="mock", python_session=True)
    server = RockyMCP(name="Ansys Rocky (python)", config=config)
    async with Client(server) as mcp_client:
        yield mcp_client


async def test_run_python_executes_code(python_client: Client) -> None:
    """Arbitrary Python runs in the persistent session."""
    result = await python_client.call_tool("rocky_run_python", {"code": "print(6 * 7)"})
    assert "42" in str(result.data)


async def test_run_python_keeps_state_between_calls(python_client: Client) -> None:
    """Variables survive from one call to the next."""
    await python_client.call_tool("rocky_run_python", {"code": "carried = 17"})
    result = await python_client.call_tool("rocky_run_python", {"code": "print(carried * 2)"})
    assert "34" in str(result.data)


async def test_run_python_reports_errors(python_client: Client) -> None:
    """A failing snippet reports the error instead of crashing the server."""
    result = await python_client.call_tool("rocky_run_python", {"code": "raise ValueError('boom')"})
    assert "boom" in str(result.data)


async def test_run_python_is_disabled_without_a_session(client: Client) -> None:
    """The tool explains how to enable the session when it is turned off."""
    with pytest.raises(ToolError, match="Python session"):
        await client.call_tool("rocky_run_python", {"code": "print(1)"})
