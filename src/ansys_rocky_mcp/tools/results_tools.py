"""Post-processing and results tools for the Rocky MCP server."""

from __future__ import annotations

from typing import Any

from fastmcp import Context
from fastmcp.exceptions import ToolError

from ansys_rocky_mcp.tools._common import (
    envelope,
    get_session,
    require_study,
    to_float_list,
)

__all__ = ["register"]


def _curve(source: Any, name: str, unit: str | None) -> tuple[list[float], list[float]]:
    """Read a curve from a particles or selection object."""
    getter = getattr(source, "GetNumpyCurve", None)
    if getter is None:
        getter = getattr(source, "GetCurve", None)
    if getter is None:
        raise ToolError(f"{type(source).__name__} exposes neither GetNumpyCurve nor GetCurve.")
    try:
        if unit is None:
            times, values = getter(name)
        else:
            times, values = getter(name, unit=unit)
    except TypeError:
        times, values = getter(name)
    except Exception as exc:
        raise ToolError(f"Could not read the curve {name!r}: {exc}") from exc
    return to_float_list(times), to_float_list(values)


async def rocky_get_curve(
    ctx: Context,
    name: str,
    unit: str | None = None,
    selection_id: str | None = None,
) -> dict[str, Any]:
    """Read a time series produced by the simulation.

    Parameters
    ----------
    name : str
        Curve name, for example ``"Particles Count"`` or
        ``"Particles Mass Flow In"``.
    unit : str | None, default: None
        Unit of the curve values. When omitted, the Rocky default is used.
    selection_id : str | None, default: None
        Identifier of a selection process. When given, the curve is read for
        that selection instead of the whole domain.

    """
    session = get_session(ctx)
    study = require_study(session)

    if selection_id:
        try:
            source = session.registry.resolve(selection_id, "selection")
        except KeyError as exc:
            raise ToolError(str(exc)) from exc
    else:
        source = study.GetParticles()

    times, values = _curve(source, name, unit)
    return envelope(
        session,
        name=name,
        unit=unit,
        selection_id=selection_id,
        number_of_points=len(values),
        times=times,
        values=values,
    )


async def rocky_list_grid_functions(ctx: Context) -> dict[str, Any]:
    """List the grid functions available for the current study."""
    session = get_session(ctx)
    study = require_study(session)
    particles = study.GetParticles()
    try:
        names = list(particles.GetGridFunctionNames())
    except Exception as exc:
        raise ToolError(f"Could not list the grid functions: {exc}") from exc
    return envelope(session, grid_functions=names)


async def rocky_get_grid_function_stats(
    ctx: Context,
    name: str,
    unit: str | None = None,
) -> dict[str, Any]:
    """Read the minimum and maximum of a grid function over the time set.

    Parameters
    ----------
    name : str
        Grid function name, for example
        ``"Velocity : Translational : Absolute"``.
    unit : str | None, default: None
        Unit of the reported values.

    """
    session = get_session(ctx)
    study = require_study(session)
    particles = study.GetParticles()
    try:
        grid_function = particles.GetGridFunction(name)
    except Exception as exc:
        raise ToolError(f"Could not open the grid function {name!r}: {exc}") from exc

    try:
        times = list(study.GetTimeSet() or [])
    except Exception:
        times = []

    series: list[dict[str, Any]] = []
    for index, time in enumerate(times):
        try:
            maximum = grid_function.GetMax(unit=unit, time_step=index)
            minimum = grid_function.GetMin(unit=unit, time_step=index)
        except TypeError:
            maximum = grid_function.GetMax(unit)
            minimum = grid_function.GetMin(unit)
        except Exception as exc:
            raise ToolError(
                f"Could not read the grid function {name!r} at time step {index}: {exc}"
            ) from exc
        series.append(
            {
                "time_step": index,
                "time_s": float(time),
                "max": float(maximum),
                "min": float(minimum),
            }
        )

    maxima = [point["max"] for point in series]
    minima = [point["min"] for point in series]
    return envelope(
        session,
        name=name,
        unit=unit,
        number_of_time_steps=len(series),
        overall_max=max(maxima) if maxima else None,
        overall_min=min(minima) if minima else None,
        series=series,
    )


async def rocky_create_selection(
    ctx: Context,
    kind: str = "cylinder",
    size: list[float] | None = None,
    center: list[float] | None = None,
    final_angle: float | None = None,
    unit: str = "m",
    name: str | None = None,
) -> dict[str, Any]:
    """Create a selection process for post-processing.

    Parameters
    ----------
    kind : str, default: "cylinder"
        Either ``"cylinder"`` or ``"cube"``.
    size : list[float] | None, default: None
        Size of the selection as ``[x, y, z]``.
    center : list[float] | None, default: None
        Center of the selection as ``[x, y, z]``.
    final_angle : float | None, default: None
        Final angle of a cylinder selection in degrees.
    unit : str, default: "m"
        Unit of the size and the center.
    name : str | None, default: None
        Name for the created selection.

    """
    session = get_session(ctx)
    study = require_study(session)
    kind = kind.strip().lower()
    if kind not in ("cylinder", "cube"):
        raise ToolError(f"Unknown selection kind {kind!r}. Expected 'cylinder' or 'cube'.")

    processes = study.GetUserProcessCollection()
    particles = study.GetParticles()
    try:
        if kind == "cylinder":
            selection = processes.CreateCylinderProcess(particles, name=name)
        else:
            selection = processes.CreateCubeProcess(particles, name=name)
    except Exception as exc:
        raise ToolError(f"Could not create the selection: {exc}") from exc

    try:
        if size is not None:
            values = list(size)
            while len(values) < 3:
                values.append(values[-1])
            selection.SetSize(values[0], values[1], values[2], unit=unit)
        if center is not None:
            values = list(center)
            while len(values) < 3:
                values.append(values[-1])
            selection.SetCenter(values[0], values[1], values[2], unit=unit)
        if final_angle is not None:
            selection.SetFinalAngle(final_angle)
    except Exception as exc:
        raise ToolError(f"Could not size the selection: {exc}") from exc

    ident = session.registry.add(
        "selection",
        selection,
        label=name or f"{kind.title()} selection",
        meta={"kind": kind},
    )
    return envelope(
        session,
        message=f"Created a {kind} selection.",
        selection_id=ident,
        kind=kind,
        name=getattr(selection, "GetName", lambda: name or "")(),
    )


async def rocky_list_processes(ctx: Context) -> dict[str, Any]:
    """List the user processes and registered objects of the study."""
    session = get_session(ctx)
    study = require_study(session)
    processes = study.GetUserProcessCollection()
    try:
        names = list(processes.GetProcessNames())
    except Exception:
        names = []
    return envelope(
        session,
        processes=names,
        registered_objects=session.registry.list(),
    )


async def rocky_run_python(ctx: Context, code: str, timeout: int = 60) -> str:
    """Run arbitrary Python code in the persistent Python session.

    This is the escape hatch for anything the dedicated tools do not cover.
    The session keeps state between calls, so variables survive from one
    invocation to the next.

    Parameters
    ----------
    code : str
        Python source to execute.
    timeout : int, default: 60
        Maximum number of seconds to wait for the code to finish.

    """
    from ansys.common.mcp.tools import execute_python_code

    get_session(ctx)
    lifespan = getattr(ctx.request_context, "lifespan_context", None)
    if getattr(lifespan, "python_session", None) is None:
        raise ToolError(
            "The persistent Python session is disabled. Restart the server without "
            "'--no-python-session' to enable it."
        )
    return await execute_python_code(ctx, code, timeout=timeout)


def register(mcp: Any) -> None:
    """Register the results tools on the MCP server."""
    mcp.tool()(rocky_get_curve)
    mcp.tool()(rocky_list_grid_functions)
    mcp.tool()(rocky_get_grid_function_stats)
    mcp.tool()(rocky_create_selection)
    mcp.tool()(rocky_list_processes)
    mcp.tool()(rocky_run_python)
