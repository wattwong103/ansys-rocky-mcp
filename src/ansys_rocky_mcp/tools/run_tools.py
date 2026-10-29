"""Simulation run-control tools for the Rocky MCP server."""

from __future__ import annotations

import asyncio
from typing import Any

from fastmcp import Context
from fastmcp.exceptions import ToolError

from ansys_rocky_mcp.tools._common import envelope, get_session, require_study

__all__ = ["register"]


async def rocky_run_simulation(
    ctx: Context,
    non_blocking: bool = True,
    delete_results: bool = False,
) -> dict[str, Any]:
    """Start the Rocky simulation.

    Parameters
    ----------
    non_blocking : bool, default: True
        When ``True`` the call returns as soon as the solver starts, so that
        progress can be polled with ``rocky_simulation_status``. When
        ``False`` the call blocks until the simulation finishes.
    delete_results : bool, default: False
        Delete any previous results and start from scratch.

    """
    session = get_session(ctx)
    study = require_study(session)
    try:
        started = study.StartSimulation(
            skip_summary=True,
            delete_results=delete_results,
            non_blocking=non_blocking,
        )
    except TypeError:
        started = study.StartSimulation(non_blocking=non_blocking)
    except Exception as exc:
        raise ToolError(f"Could not start the simulation: {exc}") from exc

    return envelope(
        session,
        message="Simulation started." if started else "The simulation was already running.",
        started=bool(started),
        non_blocking=non_blocking,
        progress_percent=study.GetProgress(),
        simulating=bool(study.IsSimulating()),
    )


async def rocky_simulation_status(ctx: Context) -> dict[str, Any]:
    """Report the state of the running simulation.

    Returns whether the solver is running, the progress percentage, the
    computed time set and whether results are available.
    """
    session = get_session(ctx)
    study = require_study(session)
    try:
        times = list(study.GetTimeSet() or [])
    except Exception:
        times = []
    return envelope(
        session,
        simulating=bool(study.IsSimulating()),
        progress_percent=study.GetProgress(),
        has_results=bool(study.HasResults()),
        time_set_s=times,
        number_of_time_steps=len(times),
    )


async def rocky_wait_for_completion(
    ctx: Context,
    timeout: float = 600.0,
    poll_interval: float = 2.0,
) -> dict[str, Any]:
    """Wait until the running simulation finishes.

    Parameters
    ----------
    timeout : float, default: 600.0
        Maximum number of seconds to wait.
    poll_interval : float, default: 2.0
        Number of seconds between two progress checks.

    """
    session = get_session(ctx)
    study = require_study(session)
    if not study.IsSimulating():
        return envelope(
            session,
            message="No simulation is running.",
            completed=True,
            timed_out=False,
            progress_percent=study.GetProgress(),
            has_results=bool(study.HasResults()),
        )

    waited = 0.0
    while study.IsSimulating() and waited < timeout:
        await asyncio.sleep(poll_interval)
        waited += poll_interval

    completed = not study.IsSimulating()
    return envelope(
        session,
        message=(
            "Simulation finished."
            if completed
            else f"Still running after {waited:.0f} s; the timeout was reached."
        ),
        completed=completed,
        timed_out=not completed,
        waited_s=waited,
        progress_percent=study.GetProgress(),
        has_results=bool(study.HasResults()),
    )


def register(mcp: Any) -> None:
    """Register the run-control tools on the MCP server."""
    mcp.tool()(rocky_run_simulation)
    mcp.tool()(rocky_simulation_status)
    mcp.tool()(rocky_wait_for_completion)
