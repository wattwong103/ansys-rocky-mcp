"""Study setup tools: geometry, particles, inlets, outlets, domain and solver."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import Context
from fastmcp.exceptions import ToolError

from ansys_rocky_mcp.tools._common import envelope, get_session, require_study

__all__ = ["register"]

SURFACE_KINDS = ("circular", "rectangular")


def _resolve(session: Any, object_id: str, category: str) -> Any:
    """Resolve a registered object id into the live Rocky object."""
    try:
        return session.registry.resolve(object_id, category)
    except KeyError as exc:
        raise ToolError(str(exc)) from exc


async def rocky_import_wall(
    ctx: Context,
    path: str,
    scale: float = 1.0,
    name_prefix: str | None = None,
) -> dict[str, Any]:
    """Import a geometry file as a Rocky wall.

    Parameters
    ----------
    path : str
        Path to the geometry file (STL, OBJ and the formats Rocky supports).
    scale : float, default: 1.0
        Import scale factor applied to the geometry.
    name_prefix : str | None, default: None
        Prefix for the created wall names.

    """
    session = get_session(ctx)
    study = require_study(session)
    target = Path(path).expanduser()
    if not target.exists():
        raise ToolError(f"Geometry file not found: {target}")
    try:
        walls = study.ImportWall(
            str(target),
            import_scale=scale,
            custom_name_prefix=name_prefix,
        )
    except TypeError:
        # Older Rocky builds do not accept every keyword argument.
        walls = study.ImportWall(str(target))
    except Exception as exc:
        raise ToolError(f"Could not import the wall {target}: {exc}") from exc

    walls = list(walls or [])
    ids = [session.registry.add("wall", wall, label=f"Wall from {target.name}") for wall in walls]
    return envelope(
        session,
        message=f"Imported {len(walls)} wall(s) from {target}.",
        wall_ids=ids,
        walls=[
            {"id": ident, "name": getattr(wall, "GetName", lambda: "")()}
            for ident, wall in zip(ids, walls, strict=True)
        ],
    )


async def rocky_create_surface(
    ctx: Context,
    kind: str = "circular",
    center: list[float] | None = None,
    radius: float | None = None,
    length: float | None = None,
    width: float | None = None,
    unit: str = "m",
    name: str | None = None,
) -> dict[str, Any]:
    """Create a circular or rectangular Rocky surface.

    Parameters
    ----------
    kind : str, default: "circular"
        Either ``"circular"`` or ``"rectangular"``.
    center : list[float] | None, default: None
        Center of the surface as ``[x, y, z]``.
    radius : float | None, default: None
        Maximum radius. Required for a circular surface.
    length : float | None, default: None
        Length. Required for a rectangular surface.
    width : float | None, default: None
        Width. Required for a rectangular surface.
    unit : str, default: "m"
        Unit of every length value.
    name : str | None, default: None
        Name for the created surface.

    """
    session = get_session(ctx)
    study = require_study(session)
    kind = kind.strip().lower()
    if kind not in SURFACE_KINDS:
        raise ToolError(f"Unknown surface kind {kind!r}. Expected one of {SURFACE_KINDS}.")

    if kind == "circular":
        surface = study.CreateCircularSurface()
    else:
        surface = study.CreateRectangularSurface()

    if center is not None:
        try:
            surface.SetCenter(list(center), unit=unit)
        except Exception as exc:
            raise ToolError(f"Could not set the surface center: {exc}") from exc

    try:
        if kind == "circular":
            if radius is not None:
                surface.SetMaxRadius(radius, unit=unit)
        else:
            if length is not None:
                surface.SetLength(length, unit=unit)
            if width is not None:
                surface.SetWidth(width, unit=unit)
    except Exception as exc:
        raise ToolError(f"Could not size the {kind} surface: {exc}") from exc

    if name:
        try:
            surface.SetName(name)
        except Exception:  # pragma: no cover - naming is best effort
            pass

    ident = session.registry.add(
        "surface",
        surface,
        label=name or f"{kind.title()} surface",
        meta={"kind": kind},
    )
    return envelope(
        session,
        message=f"Created a {kind} surface.",
        surface_id=ident,
        kind=kind,
        name=getattr(surface, "GetName", lambda: name or "")(),
    )


async def rocky_create_particle(
    ctx: Context,
    shape: str = "sphere",
    sieve_size: float | None = None,
    unit: str = "m",
    name: str | None = None,
) -> dict[str, Any]:
    """Create a Rocky particle definition.

    Parameters
    ----------
    shape : str, default: "sphere"
        Particle shape, for example ``"sphere"`` or ``"sphero_cylinder"``.
    sieve_size : float | None, default: None
        Sieve size of the particle. This sets the particle mass and therefore
        the particle throughput for a given mass flow.
    unit : str, default: "m"
        Unit of the sieve size.
    name : str | None, default: None
        Name for the created particle.

    """
    session = get_session(ctx)
    study = require_study(session)
    particle = study.CreateParticle()
    try:
        particle.SetShape(shape)
        if sieve_size is not None:
            particle.SetMinimumSize(sieve_size, unit=unit)
        if name:
            particle.SetName(name)
    except Exception as exc:
        raise ToolError(f"Could not configure the particle: {exc}") from exc

    ident = session.registry.add(
        "particle",
        particle,
        label=name or f"Particle ({shape})",
        meta={"shape": shape, "sieve_size": sieve_size, "unit": unit},
    )
    return envelope(
        session,
        message=f"Created a particle definition with shape {shape!r}.",
        particle_id=ident,
        shape=shape,
        name=getattr(particle, "GetName", lambda: name or "")(),
    )


async def rocky_create_inlet(
    ctx: Context,
    surface_id: str,
    particle_id: str,
    mass_flow_rate: float,
    unit: str = "t/h",
) -> dict[str, Any]:
    """Create a particle inlet on a surface and set its mass flow rate.

    Parameters
    ----------
    surface_id : str
        Identifier of the surface acting as the entry point.
    particle_id : str
        Identifier of the particle definition fed by the inlet.
    mass_flow_rate : float
        Mass flow rate of the inlet.
    unit : str, default: "t/h"
        Unit of the mass flow rate.

    """
    session = get_session(ctx)
    study = require_study(session)
    surface = _resolve(session, surface_id, "surface")
    particle = _resolve(session, particle_id, "particle")

    try:
        inlet = study.CreateParticleInlet(entry_point=surface, particle=particle)
    except TypeError:
        inlet = study.CreateParticleInlet(surface, particle)
    except Exception as exc:
        raise ToolError(f"Could not create the particle inlet: {exc}") from exc

    try:
        properties = inlet.GetInputPropertiesList()
        if not properties:
            raise ToolError("The created inlet exposes no input properties.")
        properties[0].SetMassFlowRate(mass_flow_rate, unit=unit)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not set the inlet mass flow rate: {exc}") from exc

    ident = session.registry.add(
        "inlet",
        inlet,
        label=f"Inlet on {surface_id}",
        meta={"surface_id": surface_id, "particle_id": particle_id},
    )
    return envelope(
        session,
        message=f"Created a particle inlet with a mass flow of {mass_flow_rate} {unit}.",
        inlet_id=ident,
        surface_id=surface_id,
        particle_id=particle_id,
        mass_flow_rate=mass_flow_rate,
        unit=unit,
    )


async def rocky_create_outlet(ctx: Context, surface_id: str) -> dict[str, Any]:
    """Create a Rocky outlet on a surface.

    Parameters
    ----------
    surface_id : str
        Identifier of the surface acting as the exit point.

    """
    session = get_session(ctx)
    study = require_study(session)
    surface = _resolve(session, surface_id, "surface")
    try:
        outlet = study.CreateOutlet(surface)
    except Exception as exc:
        raise ToolError(f"Could not create the outlet: {exc}") from exc

    ident = session.registry.add(
        "outlet",
        outlet,
        label=f"Outlet on {surface_id}",
        meta={"surface_id": surface_id},
    )
    return envelope(
        session,
        message="Created an outlet.",
        outlet_id=ident,
        surface_id=surface_id,
    )


async def rocky_set_domain(
    ctx: Context,
    min_values: list[float] | None = None,
    max_values: list[float] | None = None,
    use_boundary_limits: bool | None = None,
    unit: str = "m",
) -> dict[str, Any]:
    """Configure the Rocky domain limits.

    Parameters
    ----------
    min_values : list[float] | None, default: None
        Minimum domain coordinates as ``[x, y, z]``.
    max_values : list[float] | None, default: None
        Maximum domain coordinates as ``[x, y, z]``.
    use_boundary_limits : bool | None, default: None
        Whether Rocky uses the boundary limits of the imported geometry.
    unit : str, default: "m"
        Unit of the domain coordinates.

    """
    session = get_session(ctx)
    study = require_study(session)
    domain = study.GetDomainSettings()
    try:
        if use_boundary_limits is True:
            domain.EnableUseBoundaryLimits()
        elif use_boundary_limits is False:
            domain.DisableUseBoundaryLimits()
        if min_values is not None:
            domain.SetCoordinateLimitsMinValues(list(min_values), unit=unit)
        if max_values is not None:
            domain.SetCoordinateLimitsMaxValues(list(max_values), unit=unit)
    except Exception as exc:
        raise ToolError(f"Could not configure the domain: {exc}") from exc

    return envelope(
        session,
        message="Domain settings updated.",
        min_m=list(domain.GetCoordinateLimitsMinValues("m")),
        max_m=list(domain.GetCoordinateLimitsMaxValues("m")),
        use_boundary_limits=bool(domain.GetUseBoundaryLimits()),
    )


async def rocky_set_simulation_settings(
    ctx: Context,
    duration: float | None = None,
    time_interval: float | None = None,
    processors: int | None = None,
    execution_mode: str | None = None,
    unit: str = "s",
) -> dict[str, Any]:
    """Configure the Rocky solver settings.

    Parameters
    ----------
    duration : float | None, default: None
        Total simulation duration.
    time_interval : float | None, default: None
        Interval between the stored output time steps.
    processors : int | None, default: None
        Number of processors used by the solver.
    execution_mode : str | None, default: None
        Execution mode, for example ``"parallel"`` or ``"serial"``.
    unit : str, default: "s"
        Unit of the duration and the time interval.

    """
    session = get_session(ctx)
    study = require_study(session)
    solver = study.GetSolver()
    try:
        if duration is not None:
            solver.SetSimulationDuration(duration, unit=unit)
        if time_interval is not None:
            solver.SetTimeInterval(time_interval, unit=unit)
        if processors is not None:
            solver.SetNumberOfProcessors(int(processors))
        if execution_mode is not None:
            solver.SetExecutionMode(execution_mode)
    except Exception as exc:
        raise ToolError(f"Could not configure the solver: {exc}") from exc

    return envelope(
        session,
        message="Solver settings updated.",
        duration_s=solver.GetSimulationDuration("s"),
        time_interval_s=solver.GetTimeInterval("s"),
        processors=solver.GetNumberOfProcessors(),
        execution_mode=solver.GetExecutionMode(),
    )


async def rocky_set_motion(
    ctx: Context,
    kind: str,
    targets: list[str],
    angular_velocity: list[float] | None = None,
    linear_velocity: list[float] | None = None,
    unit: str = "rad/s",
    start_time: float = 0.0,
    stop_time: float = 1000.0,
) -> dict[str, Any]:
    """Apply a rotation or translation motion to one or more study objects.

    Parameters
    ----------
    kind : str
        Either ``"rotation"`` or ``"translation"``.
    targets : list[str]
        Identifiers of the objects the motion is applied to, for example
        ``["surface_1"]``.
    angular_velocity : list[float] | None, default: None
        Angular velocity vector as ``[wx, wy, wz]``. Required for a rotation.
    linear_velocity : list[float] | None, default: None
        Linear velocity vector as ``[vx, vy, vz]``. Required for a translation.
    unit : str, default: "rad/s"
        Unit of the angular velocity, for example ``"rad/s"`` or ``"deg/s"``.
        Translations always use ``"m/s"``.
    start_time : float, default: 0.0
        Start time of the motion in seconds.
    stop_time : float, default: 1000.0
        Stop time of the motion in seconds.

    """
    session = get_session(ctx)
    study = require_study(session)
    kind = kind.strip().lower()
    if kind not in ("rotation", "translation"):
        raise ToolError(f"Unknown motion kind {kind!r}. Expected 'rotation' or 'translation'.")
    if not targets:
        raise ToolError("At least one target object id is required.")

    try:
        source = study.GetMotionFrameSource()
        frame = source.NewFrame()
    except Exception as exc:
        raise ToolError(f"Could not create a motion frame: {exc}") from exc

    try:
        if kind == "rotation":
            if angular_velocity is None:
                raise ToolError("A rotation requires 'angular_velocity'.")
            frame.AddRotationMotion(
                start_time=(start_time, "s"),
                stop_time=(stop_time, "s"),
                angular_velocity=(tuple(angular_velocity), unit),
            )
        else:
            if linear_velocity is None:
                raise ToolError("A translation requires 'linear_velocity'.")
            frame.AddTranslationMotion(
                start_time=(start_time, "s"),
                stop_time=(stop_time, "s"),
                velocity=(tuple(linear_velocity), "m/s"),
            )
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not define the motion: {exc}") from exc

    applied: list[str] = []
    for target_id in targets:
        obj = _resolve(session, target_id, None)
        try:
            frame.ApplyTo(obj)
        except Exception as exc:
            raise ToolError(f"Could not apply the motion to {target_id}: {exc}") from exc
        applied.append(target_id)

    ident = session.registry.add(
        "motion",
        frame,
        label=f"{kind} motion",
        meta={"kind": kind, "targets": applied},
    )
    return envelope(
        session,
        message=f"Applied a {kind} motion to {len(applied)} object(s).",
        motion_id=ident,
        kind=kind,
        targets=applied,
    )


def register(mcp: Any) -> None:
    """Register the setup tools on the MCP server."""
    mcp.tool()(rocky_import_wall)
    mcp.tool()(rocky_create_surface)
    mcp.tool()(rocky_create_particle)
    mcp.tool()(rocky_create_inlet)
    mcp.tool()(rocky_create_outlet)
    mcp.tool()(rocky_set_domain)
    mcp.tool()(rocky_set_simulation_settings)
    mcp.tool()(rocky_set_motion)
