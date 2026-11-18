"""Study setup tools: geometry, particles, inlets, outlets, domain and solver."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from fastmcp import Context
from fastmcp.exceptions import ToolError

from ansys_rocky_mcp.tools._common import envelope, get_session, require_study

__all__ = ["register"]

SURFACE_KINDS = ("circular", "rectangular")
DOMAIN_TYPES = ("NONE", "CARTESIAN", "CYLINDRICAL")
CARTESIAN_PERIODIC = ("NONE", "X", "Y", "XY", "Z", "XZ", "YZ", "XYZ")
CYLINDRICAL_PERIODIC = ("X", "Y", "Z")
EXECUTION_MODES = ("local", "single_mpi", "multi_mpi")
SIZE_DISTRIBUTIONS = ("gaudin_schumann", "incomplete_beta", "custom")
PARTICLE_SHAPES = (
    "sphere",
    "polyhedron",
    "sphero_cylinder",
    "sphero_polygon",
    "sphero_polyhedron",
    "briquete",
    "faceted_cylinder",
    "assembly",
    "straight_fiber",
    "custom_fiber",
    "custom_shell",
    "custom_polyhedron",
)
NORMAL_FORCE_MODELS = (
    "linear_hysteresis",
    "linear_elastic_viscous",
    "damped_hertzian",
    "custom",
)
TANGENTIAL_FORCE_MODELS = (
    "elastic_coulomb",
    "coulomb_limit",
    "mindlin_deresiewicz",
    "custom",
)
ROLLING_RESISTANCE_MODELS = ("type_1", "type_3", "none", "custom")
ADHESION_MODELS = ("none", "constant", "linear", "JKR", "custom")
RESTITUTION_MODELS = ("constant", "velocity_dependent")


def _angular_velocity_rad_s(values: list[float], unit: str) -> tuple[float, float, float]:
    """Return an angular velocity in rad/s. Degrees per second are converted."""
    if len(values) != 3:
        raise ToolError("angular_velocity must be three numbers.")
    if unit == "rad/s":
        scale = 1.0
    elif unit in ("deg/s", "dega/s"):
        scale = math.pi / 180.0
    else:
        raise ToolError(
            f"Unknown angular velocity unit {unit!r}. Expected 'rad/s', 'deg/s', or 'dega/s'."
        )
    return tuple(float(value) * scale for value in values)  # type: ignore[return-value]


def _modulus_unit(unit: str) -> str:
    """Return Rocky's Young's modulus token. ``Pa`` is the same unit as ``N/m2``."""
    if unit in ("N/m2", "Pa", "N/m^2"):
        return "N/m2"
    raise ToolError(f"Unknown Young's modulus unit {unit!r}. Expected 'N/m2' or 'Pa'.")


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


async def rocky_create_feed_conveyor(
    ctx: Context,
    belt_speed: float = 1.0,
    belt_width: float = 1.0,
    incline_deg: float = 0.0,
    start_time: float | None = None,
    acceleration_period: float | None = None,
    stop_time: float | None = None,
    deceleration_period: float | None = None,
    belt_profile: str | None = None,
    loading_length: float | None = None,
    drop_length: float | None = None,
    drop_width: float | None = None,
    drop_height: float | None = None,
    skirtboard_height: float | None = None,
    transition_length: float | None = None,
    triangle_size: float | None = None,
    belt_thickness: float | None = None,
    idler_diameter: float | None = None,
    wall_thickness: float | None = None,
    face_width: float | None = None,
    alignment_deg: float | None = None,
    horizontal_offset: float | None = None,
    vertical_offset: float | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Create a Rocky feed conveyor.

    Parameters
    ----------
    belt_speed : float, default: 1.0
        Belt speed in meters per second.
    belt_width : float, default: 1.0
        Belt width in meters.
    incline_deg : float, default: 0.0
        Belt incline in degrees.
    start_time : float | None, default: None
        Time in seconds when the belt starts moving.
    acceleration_period : float | None, default: None
        Seconds the belt takes to reach ``belt_speed``.
    stop_time : float | None, default: None
        Time in seconds when the belt starts to stop.
    deceleration_period : float | None, default: None
        Seconds the belt takes to stop.
    belt_profile : str | None, default: None
        Belt profile name. Rocky's object lists the accepted names.
    loading_length : float | None, default: None
        Length of the loading section in meters. Feed conveyors only.
    drop_length : float | None, default: None
        Drop box length in meters. Feed conveyors only.
    drop_width : float | None, default: None
        Drop box width in meters. Feed conveyors only.
    drop_height : float | None, default: None
        Drop box height in meters. Feed conveyors only.
    skirtboard_height : float | None, default: None
        Skirtboard height in meters. Feed conveyors only.
    transition_length : float | None, default: None
        Transition length in meters. Feed conveyors only.
    triangle_size : float | None, default: None
        Belt mesh triangle size in meters.
    belt_thickness : float | None, default: None
        Belt thickness in meters.
    idler_diameter : float | None, default: None
        Pulley diameter in meters. Feed conveyors only.
    wall_thickness : float | None, default: None
        Side wall thickness in meters. Feed conveyors only.
    face_width : float | None, default: None
        Pulley face width in meters. Feed conveyors only.
    alignment_deg : float | None, default: None
        Alignment angle in degrees. Rocky stores this angle as ``dega``.
    horizontal_offset : float | None, default: None
        Horizontal offset in meters.
    vertical_offset : float | None, default: None
        Vertical offset in meters.
    name : str | None, default: None
        Conveyor name.

    """
    return await _create_conveyor(
        ctx,
        "feed",
        belt_speed,
        belt_width,
        incline_deg,
        name,
        start_time=start_time,
        acceleration_period=acceleration_period,
        stop_time=stop_time,
        deceleration_period=deceleration_period,
        belt_profile=belt_profile,
        loading_length=loading_length,
        drop_length=drop_length,
        drop_width=drop_width,
        drop_height=drop_height,
        skirtboard_height=skirtboard_height,
        transition_length=transition_length,
        triangle_size=triangle_size,
        belt_thickness=belt_thickness,
        idler_diameter=idler_diameter,
        wall_thickness=wall_thickness,
        face_width=face_width,
        alignment_deg=alignment_deg,
        horizontal_offset=horizontal_offset,
        vertical_offset=vertical_offset,
    )


async def _create_conveyor(
    ctx: Context,
    kind: str,
    belt_speed: float,
    belt_width: float,
    incline_deg: float,
    name: str | None,
    start_time: float | None = None,
    acceleration_period: float | None = None,
    stop_time: float | None = None,
    deceleration_period: float | None = None,
    belt_profile: str | None = None,
    loading_length: float | None = None,
    drop_length: float | None = None,
    drop_width: float | None = None,
    drop_height: float | None = None,
    skirtboard_height: float | None = None,
    transition_length: float | None = None,
    triangle_size: float | None = None,
    belt_thickness: float | None = None,
    idler_diameter: float | None = None,
    wall_thickness: float | None = None,
    face_width: float | None = None,
    alignment_deg: float | None = None,
    horizontal_offset: float | None = None,
    vertical_offset: float | None = None,
) -> dict[str, Any]:
    """Create a feed or receiving conveyor."""
    session = get_session(ctx)
    study = require_study(session)
    try:
        if kind == "receiving":
            conveyor = study.CreateReceivingConveyor()
        else:
            conveyor = study.CreateFeedConveyor()
        conveyor.SetBeltSpeed(belt_speed, unit="m/s")
        conveyor.SetBeltWidth(belt_width, unit="m")
        conveyor.SetBeltInclineAngle(incline_deg, unit="dega")
        if start_time is not None:
            conveyor.SetBeginningStartTime(start_time, unit="s")
        if acceleration_period is not None:
            conveyor.SetAccelerationPeriod(acceleration_period, unit="s")
        if stop_time is not None:
            conveyor.SetBeginningStopTime(stop_time, unit="s")
        if deceleration_period is not None:
            conveyor.SetDecelerationPeriod(deceleration_period, unit="s")
        if belt_profile is not None:
            valid = list(conveyor.GetValidBeltProfileNames())
            if belt_profile not in valid:
                raise ToolError(f"Unknown belt profile {belt_profile!r}. Expected one of {valid}.")
            conveyor.SetBeltProfile(belt_profile)
        if name:
            conveyor.SetName(name)
        if loading_length is not None:
            conveyor.SetLoadingLength(loading_length, unit="m")
        if drop_length is not None:
            conveyor.SetDropBoxLength(drop_length, unit="m")
        if drop_width is not None:
            conveyor.SetDropBoxWidth(drop_width, unit="m")
        if drop_height is not None:
            conveyor.SetDropBoxHeight(drop_height, unit="m")
        if skirtboard_height is not None:
            conveyor.SetSkirtboardHeight(skirtboard_height, unit="m")
        if transition_length is not None:
            conveyor.SetTransitionLength(transition_length, unit="m")
        if triangle_size is not None:
            conveyor.SetTriangleSize(triangle_size, unit="m")
        if belt_thickness is not None:
            conveyor.SetBeltThickness(belt_thickness, unit="m")
        if idler_diameter is not None:
            conveyor.SetDiameter(idler_diameter, unit="m")
        if wall_thickness is not None:
            conveyor.SetWallThickness(wall_thickness, unit="m")
        if face_width is not None:
            conveyor.SetFaceWidth(face_width, unit="m")
        if alignment_deg is not None:
            conveyor.SetAlignmentAngle(alignment_deg, unit="dega")
        if horizontal_offset is not None:
            conveyor.SetHorizontalOffset(horizontal_offset, unit="m")
        if vertical_offset is not None:
            conveyor.SetVerticalOffset(vertical_offset, unit="m")
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not create the {kind} conveyor: {exc}") from exc
    ident = session.registry.add(
        "conveyor",
        conveyor,
        label=name or conveyor.GetName(),
        meta={"belt_speed_m_s": belt_speed},
    )
    return envelope(
        session,
        message=f"Created {kind} conveyor {conveyor.GetName()}.",
        kind=kind,
        conveyor_id=ident,
        name=conveyor.GetName(),
        belt_speed_m_s=conveyor.GetBeltSpeed("m/s"),
        belt_width_m=conveyor.GetBeltWidth("m"),
        incline_deg=conveyor.GetBeltInclineAngle("dega"),
        alignment_deg=conveyor.GetAlignmentAngle("dega"),
        start_time_s=conveyor.GetBeginningStartTime("s"),
        acceleration_period_s=conveyor.GetAccelerationPeriod("s"),
        stop_time_s=conveyor.GetBeginningStopTime("s"),
        deceleration_period_s=conveyor.GetDecelerationPeriod("s"),
        belt_profile=conveyor.GetBeltProfileName(),
        loading_length_m=conveyor.GetLoadingLength("m"),
        drop_box_m=[
            conveyor.GetDropBoxLength("m"),
            conveyor.GetDropBoxWidth("m"),
            conveyor.GetDropBoxHeight("m"),
        ],
        skirtboard_height_m=conveyor.GetSkirtboardHeight("m"),
        transition_length_m=conveyor.GetTransitionLength("m"),
        triangle_size_m=conveyor.GetTriangleSize("m"),
        belt_thickness_m=conveyor.GetBeltThickness("m"),
        idler_diameter_m=conveyor.GetDiameter("m"),
        wall_thickness_m=conveyor.GetWallThickness("m"),
        face_width_m=conveyor.GetFaceWidth("m"),
        horizontal_offset_m=conveyor.GetHorizontalOffset("m"),
        vertical_offset_m=conveyor.GetVerticalOffset("m"),
    )


async def rocky_create_receiving_conveyor(
    ctx: Context,
    belt_speed: float = 1.0,
    belt_width: float = 1.0,
    incline_deg: float = 0.0,
    start_time: float | None = None,
    acceleration_period: float | None = None,
    stop_time: float | None = None,
    deceleration_period: float | None = None,
    belt_profile: str | None = None,
    triangle_size: float | None = None,
    belt_thickness: float | None = None,
    alignment_deg: float | None = None,
    horizontal_offset: float | None = None,
    vertical_offset: float | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Create a Rocky receiving conveyor.

    Parameters match ``rocky_create_feed_conveyor``.
    """
    return await _create_conveyor(
        ctx,
        "receiving",
        belt_speed,
        belt_width,
        incline_deg,
        name,
        start_time=start_time,
        acceleration_period=acceleration_period,
        stop_time=stop_time,
        deceleration_period=deceleration_period,
        belt_profile=belt_profile,
        triangle_size=triangle_size,
        belt_thickness=belt_thickness,
        alignment_deg=alignment_deg,
        horizontal_offset=horizontal_offset,
        vertical_offset=vertical_offset,
    )


async def rocky_create_particle(
    ctx: Context,
    shape: str = "sphere",
    sieve_size: float | None = None,
    unit: str = "m",
    horizontal_aspect_ratio: float | None = None,
    vertical_aspect_ratio: float | None = None,
    thickness: float | None = None,
    number_of_corners: int | None = None,
    edge_angle_deg: float | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    """Create a Rocky particle definition.

    Parameters
    ----------
    shape : str, default: "sphere"
        One of Rocky's particle shapes, for example ``sphere`` or ``sphero_cylinder``.
    sieve_size : float | None, default: None
        Sieve size of the particle. This sets the particle mass and therefore
        the particle throughput for a given mass flow.
    unit : str, default: "m"
        Unit of the sieve size and of ``thickness``.
    horizontal_aspect_ratio : float | None, default: None
        Horizontal aspect ratio of a non-spherical particle.
    vertical_aspect_ratio : float | None, default: None
        Vertical aspect ratio of a non-spherical particle.
    thickness : float | None, default: None
        Particle thickness. Rocky stores this in meters.
    number_of_corners : int | None, default: None
        Number of corners for a faceted particle.
    edge_angle_deg : float | None, default: None
        Edge angle in degrees. Rocky's token is ``dega``.
    name : str | None, default: None
        Name for the created particle.

    """
    if shape not in PARTICLE_SHAPES:
        raise ToolError(f"Unknown particle shape {shape!r}. Expected one of {PARTICLE_SHAPES}.")
    session = get_session(ctx)
    study = require_study(session)
    particle = study.CreateParticle()
    try:
        particle.SetShape(shape)
        if sieve_size is not None:
            particle.SetMinimumSize(sieve_size, unit=unit)
        if horizontal_aspect_ratio is not None:
            particle.SetHorizontalAspectRatio(horizontal_aspect_ratio)
        if vertical_aspect_ratio is not None:
            particle.SetVerticalAspectRatio(vertical_aspect_ratio)
        if thickness is not None:
            particle.SetThickness(thickness, unit="m")
        if number_of_corners is not None:
            particle.SetNumberOfCorners(number_of_corners)
        if edge_angle_deg is not None:
            particle.SetEdgeAngle(edge_angle_deg, unit="dega")
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
        horizontal_aspect_ratio=particle.GetHorizontalAspectRatio(),
        vertical_aspect_ratio=particle.GetVerticalAspectRatio(),
        thickness_m=particle.GetThickness("m"),
        number_of_corners=particle.GetNumberOfCorners(),
        edge_angle_deg=particle.GetEdgeAngle("dega"),
    )


async def rocky_set_size_distribution(
    ctx: Context,
    particle_id: str,
    model: str,
    minimum_size: float | None = None,
    surface_energy: float | None = None,
    unit: str = "m",
    energy_unit: str = "J/m2",
) -> dict[str, Any]:
    """Set the size distribution model on a particle.

    Parameters
    ----------
    particle_id : str
        Particle identifier, for example ``particle_1``.
    model : str
        One of ``gaudin_schumann``, ``incomplete_beta``, or ``custom``.
    minimum_size : float | None, default: None
        Minimum sieve size. Omit it to leave the current size.
    unit : str, default: "m"
        Unit of ``minimum_size``.
    surface_energy : float | None, default: None
        Surface energy used by an adhesion model such as JKR.
    energy_unit : str, default: "J/m2"
        Unit of ``surface_energy``.

    """
    session = get_session(ctx)
    require_study(session)
    if model not in SIZE_DISTRIBUTIONS:
        raise ToolError(
            f"Unknown size distribution {model!r}. Expected one of {SIZE_DISTRIBUTIONS}."
        )
    particle = _resolve(session, particle_id, "particle")
    try:
        particle.SetDistributionModel(model)
        if minimum_size is not None:
            particle.SetMinimumSize(minimum_size, unit=unit)
        if surface_energy is not None:
            particle.SetSurfaceEnergy(surface_energy, unit=energy_unit)
    except Exception as exc:
        raise ToolError(f"Could not set the size distribution: {exc}") from exc
    return envelope(
        session,
        message="Size distribution updated.",
        particle_id=particle_id,
        model=particle.GetDistributionModel(),
        minimum_size_m=particle.GetMinimumSize("m"),
        surface_energy=particle.GetSurfaceEnergy("J/m2"),
    )


async def rocky_create_inlet(
    ctx: Context,
    particle_id: str,
    mass_flow_rate: float,
    surface_id: str | None = None,
    conveyor_id: str | None = None,
    unit: str = "t/h",
    injection_duration: float | None = None,
    duration_unit: str = "s",
) -> dict[str, Any]:
    """Create a particle inlet on a surface or a feed conveyor.

    Parameters
    ----------
    particle_id : str
        Identifier of the particle definition fed by the inlet.
    mass_flow_rate : float
        Mass flow rate of the inlet.
    surface_id : str | None, default: None
        Identifier of the surface acting as the entry point.
    conveyor_id : str | None, default: None
        Identifier of the feed conveyor acting as the entry point.
        Pass this instead of ``surface_id``.
    unit : str, default: "t/h"
        Unit of the mass flow rate.
    injection_duration : float | None, default: None
        How long the inlet feeds particles. Omit it to leave Rocky's duration.
    duration_unit : str, default: "s"
        Unit of ``injection_duration``.

    """
    session = get_session(ctx)
    study = require_study(session)
    if (surface_id is None) == (conveyor_id is None):
        raise ToolError("Pass either surface_id or conveyor_id, not both and not neither.")
    if conveyor_id is not None:
        entry_point = _resolve(session, conveyor_id, "conveyor")
        label = f"Inlet on {conveyor_id}"
    else:
        entry_point = _resolve(session, str(surface_id), "surface")
        label = f"Inlet on {surface_id}"
    particle = _resolve(session, particle_id, "particle")

    try:
        inlet = study.CreateParticleInlet(entry_point=entry_point, particle=particle)
    except TypeError:
        inlet = study.CreateParticleInlet(entry_point, particle)
    except Exception as exc:
        raise ToolError(f"Could not create the particle inlet: {exc}") from exc

    try:
        properties = inlet.GetInputPropertiesList()
        if not properties:
            raise ToolError("The created inlet exposes no input properties.")
        properties[0].SetMassFlowRate(mass_flow_rate, unit=unit)
        if injection_duration is not None:
            inlet.SetInjectionDuration(injection_duration, unit=duration_unit)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not set the inlet mass flow rate: {exc}") from exc

    ident = session.registry.add(
        "inlet",
        inlet,
        label=label,
        meta={"surface_id": surface_id, "conveyor_id": conveyor_id, "particle_id": particle_id},
    )
    return envelope(
        session,
        message=f"Created a particle inlet with a mass flow of {mass_flow_rate} {unit}.",
        inlet_id=ident,
        surface_id=surface_id,
        conveyor_id=conveyor_id,
        particle_id=particle_id,
        mass_flow_rate=mass_flow_rate,
        unit=unit,
        injection_duration_s=inlet.GetInjectionDuration("s"),
    )


def _three_vector(values: list[float] | None, label: str) -> list[float] | None:
    """Require a 3-number sequence, or accept an omitted value."""
    if values is None:
        return None
    if len(values) != 3:
        raise ToolError(f"{label} must be three numbers.")
    return [float(value) for value in values]


async def rocky_create_volumetric_inlet(
    ctx: Context,
    particle_id: str,
    name: str | None = None,
    mass: float = 100.0,
    seed: list[float] | None = None,
    geometry_ids: list[str] | None = None,
    use_geometries_to_compute: bool = False,
    box_center: list[float] | None = None,
    box_dimensions: list[float] | None = None,
    use_box_center_as_seed_point: bool = False,
) -> dict[str, Any]:
    """Fill a region with particles at the start of the simulation.

    Parameters
    ----------
    particle_id : str
        Identifier of the particle definition to place.
    name : str | None, default: None
        Name of the volumetric inlet.
    mass : float, default: 100.0
        Mass of particles to place, in kilograms.
    seed : list[float] | None, default: None
        Seed point in meters. Three numbers.
    geometry_ids : list[str] | None, default: None
        Wall identifiers that bound the fill when
        ``use_geometries_to_compute`` is true.
    use_geometries_to_compute : bool, default: False
        When true, the selected walls bound the fill. When false, a box does.
    box_center : list[float] | None, default: None
        Center of the limit box in meters. Required for a box fill.
    box_dimensions : list[float] | None, default: None
        Length, width, and height of the limit box in meters.
    use_box_center_as_seed_point : bool, default: False
        Use the box center as the seed point.

    """
    session = get_session(ctx)
    study = require_study(session)
    particle = _resolve(session, particle_id, "particle")
    seed_coordinates = _three_vector(seed, "seed")
    center = _three_vector(box_center, "box_center")
    dimensions = _three_vector(box_dimensions, "box_dimensions")
    walls: list[Any] = []
    if use_geometries_to_compute:
        if not geometry_ids:
            raise ToolError("A geometry fill requires at least one geometry id.")
        walls = [_resolve(session, geometry_id, "wall") for geometry_id in geometry_ids]
    elif center is None or dimensions is None:
        raise ToolError("A box fill requires box_center and box_dimensions.")
    try:
        inlet = study.CreateVolumetricInlet(
            particle=particle,
            name=name,
            mass=mass,
            seed_coordinates=seed_coordinates,
            geometries=walls or None,
            use_geometries_to_compute=use_geometries_to_compute,
            box_center=center,
            box_dimensions=dimensions,
            use_box_center_as_seed_point=use_box_center_as_seed_point,
        )
        if name:
            inlet.SetName(name)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not create the volumetric inlet: {exc}") from exc
    ident = session.registry.add(
        "inlet",
        inlet,
        label=inlet.GetName(),
        meta={"particle_id": particle_id, "mass_kg": mass},
    )
    return envelope(
        session,
        message=f"Created volumetric inlet {inlet.GetName()}.",
        inlet_id=ident,
        name=inlet.GetName(),
        mass_kg=mass,
        fill="geometries" if use_geometries_to_compute else "box",
        particle_id=particle_id,
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
    domain_type: str | None = None,
    periodic_directions: str | None = None,
    number_of_divisions: int | None = None,
    periodic_min: list[float] | None = None,
    periodic_max: list[float] | None = None,
    periodic_at_geometry_limits: bool | None = None,
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
    domain_type : str | None, default: None
        One of ``NONE``, ``CARTESIAN``, or ``CYLINDRICAL``.
    periodic_directions : str | None, default: None
        For Cartesian: ``NONE``, ``X``, ``Y``, ``XY``, ``Z``, ``XZ``, ``YZ``, or ``XYZ``.
        For cylindrical: ``X``, ``Y``, or ``Z``.
    number_of_divisions : int | None, default: None
        Number of divisions for a cylindrical periodic domain. Must be at least 1.
    periodic_min : list[float] | None, default: None
        Minimum periodic boundary coordinates as ``[x, y, z]``.
    periodic_max : list[float] | None, default: None
        Maximum periodic boundary coordinates as ``[x, y, z]``.
    periodic_at_geometry_limits : bool | None, default: None
        When true, Rocky takes the periodic limits from the geometry.
    unit : str, default: "m"
        Unit of the domain coordinates.

    """
    session = get_session(ctx)
    study = require_study(session)
    if number_of_divisions is not None and number_of_divisions < 1:
        raise ToolError("number_of_divisions must be at least 1.")
    if domain_type is not None and domain_type not in DOMAIN_TYPES:
        raise ToolError(f"Unknown domain type {domain_type!r}. Expected one of {DOMAIN_TYPES}.")
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
        if domain_type is not None:
            domain.SetDomainType(domain_type)
        if periodic_directions is not None:
            kind = domain.GetDomainType()
            if kind == "CYLINDRICAL":
                if periodic_directions not in CYLINDRICAL_PERIODIC:
                    raise ToolError(
                        f"Unknown cylindrical direction {periodic_directions!r}. "
                        f"Expected one of {CYLINDRICAL_PERIODIC}."
                    )
                domain.SetCylindricalPeriodicDirections(periodic_directions)
            else:
                if periodic_directions not in CARTESIAN_PERIODIC:
                    raise ToolError(
                        f"Unknown Cartesian direction {periodic_directions!r}. "
                        f"Expected one of {CARTESIAN_PERIODIC}."
                    )
                domain.SetCartesianPeriodicDirections(periodic_directions)
        if number_of_divisions is not None:
            domain.SetNumberOfDivisions(int(number_of_divisions))
        if periodic_min is not None:
            domain.SetPeriodicLimitsMinCoordinates(
                _three_vector(periodic_min, "periodic_min"), unit=unit
            )
        if periodic_max is not None:
            domain.SetPeriodicLimitsMaxCoordinates(
                _three_vector(periodic_max, "periodic_max"), unit=unit
            )
        if periodic_at_geometry_limits is not None:
            domain.SetPeriodicAtGeometryLimits(periodic_at_geometry_limits)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not configure the domain: {exc}") from exc

    return envelope(
        session,
        message="Domain settings updated.",
        min_m=list(domain.GetCoordinateLimitsMinValues("m")),
        max_m=list(domain.GetCoordinateLimitsMaxValues("m")),
        use_boundary_limits=bool(domain.GetUseBoundaryLimits()),
        domain_type=domain.GetDomainType(),
        cartesian_periodic_directions=domain.GetCartesianPeriodicDirections(),
        cylindrical_periodic_directions=domain.GetCylindricalPeriodicDirections(),
        number_of_divisions=domain.GetNumberOfDivisions(),
        periodic_min_m=list(domain.GetPeriodicLimitsMinCoordinates("m")),
        periodic_max_m=list(domain.GetPeriodicLimitsMaxCoordinates("m")),
        periodic_at_geometry_limits=bool(domain.GetPeriodicAtGeometryLimits()),
    )


async def rocky_set_simulation_settings(
    ctx: Context,
    duration: float | None = None,
    time_interval: float | None = None,
    processors: int | None = None,
    execution_mode: str | None = None,
    curves_frequency: int | None = None,
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
        Execution mode: ``"local"``, ``"single_mpi"``, or ``"multi_mpi"``.
    curves_frequency : int | None, default: None
        How often solver curves are stored. Must be at least 1.
    unit : str, default: "s"
        Unit of the duration and the time interval.

    """
    session = get_session(ctx)
    study = require_study(session)
    if execution_mode is not None and execution_mode not in EXECUTION_MODES:
        legal = ", ".join(EXECUTION_MODES)
        raise ToolError(f"Unknown execution mode {execution_mode!r}. Expected one of {legal}.")
    if curves_frequency is not None and curves_frequency < 1:
        raise ToolError("curves_frequency must be at least 1.")
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
        if curves_frequency is not None:
            solver.SetSolverCurvesFrequency(int(curves_frequency))
    except Exception as exc:
        raise ToolError(f"Could not configure the solver: {exc}") from exc

    return envelope(
        session,
        message="Solver settings updated.",
        duration_s=solver.GetSimulationDuration("s"),
        time_interval_s=solver.GetTimeInterval("s"),
        processors=solver.GetNumberOfProcessors(),
        execution_mode=solver.GetExecutionMode(),
        curves_frequency=solver.GetSolverCurvesFrequency(),
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
        Unit of the angular velocity. Rocky stores ``rad/s``. ``deg/s`` and
        ``dega/s`` are converted before the call. Translations always use ``m/s``.
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

    radians: tuple[float, float, float] | None = None
    try:
        if kind == "rotation":
            if angular_velocity is None:
                raise ToolError("A rotation requires 'angular_velocity'.")
            radians = _angular_velocity_rad_s(angular_velocity, unit)
            frame.AddRotationMotion(
                start_time=(start_time, "s"),
                stop_time=(stop_time, "s"),
                angular_velocity=(radians, "rad/s"),
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
        angular_velocity_rad_s=list(radians) if kind == "rotation" else None,
    )


async def rocky_set_gravity(
    ctx: Context,
    gravity: list[float],
    unit: str = "m/s2",
) -> dict[str, Any]:
    """Set the gravity vector on the study physics.

    Parameters
    ----------
    gravity : list[float]
        Gravity as ``[x, y, z]``.
    unit : str, default: "m/s2"
        Unit of the gravity components.

    """
    session = get_session(ctx)
    study = require_study(session)
    if len(gravity) != 3:
        raise ToolError("Gravity requires three components [x, y, z].")
    try:
        physics = study.GetPhysics()
        physics.SetGravityXDirection(gravity[0], unit=unit)
        physics.SetGravityYDirection(gravity[1], unit=unit)
        physics.SetGravityZDirection(gravity[2], unit=unit)
        applied = [
            physics.GetGravityXDirection("m/s2"),
            physics.GetGravityYDirection("m/s2"),
            physics.GetGravityZDirection("m/s2"),
        ]
    except Exception as exc:
        raise ToolError(f"Could not set gravity: {exc}") from exc
    return envelope(session, message="Gravity updated.", gravity_m_s2=applied, unit=unit)


async def rocky_create_material(
    ctx: Context,
    name: str | None = None,
    density: float | None = None,
    youngs_modulus: float | None = None,
    poisson_ratio: float | None = None,
    specific_heat: float | None = None,
    thermal_conductivity: float | None = None,
    particle_id: str | None = None,
    wall_id: str | None = None,
    conveyor_id: str | None = None,
    density_unit: str = "kg/m3",
    modulus_unit: str = "N/m2",
) -> dict[str, Any]:
    """Create a solid material and optionally assign it to a particle, wall, or conveyor.

    Parameters
    ----------
    name : str | None, default: None
        Material name. Rocky chooses one when this is omitted.
    density : float | None, default: None
        Solid density.
    youngs_modulus : float | None, default: None
        Young's modulus.
    poisson_ratio : float | None, default: None
        Poisson's ratio.
    specific_heat : float | None, default: None
        Specific heat in ``J/kg.K``, Rocky's token.
    thermal_conductivity : float | None, default: None
        Thermal conductivity in ``W/m.K``, Rocky's token.
    particle_id : str | None, default: None
        Particle to assign the material to, for example ``particle_1``.
    wall_id : str | None, default: None
        Wall to assign the material to, for example ``wall_1``.
    conveyor_id : str | None, default: None
        Conveyor to assign the material to, for example ``conveyor_1``.
    density_unit : str, default: "kg/m3"
        Unit of ``density``.
    modulus_unit : str, default: "N/m2"
        Unit of ``youngs_modulus``. Rocky's token is ``N/m2``. ``Pa`` is accepted
        and sent as ``N/m2``.

    """
    session = get_session(ctx)
    study = require_study(session)
    try:
        material = study.GetMaterialCollection().AddSolidMaterial(name)
        if density is not None:
            material.SetDensity(density, unit=density_unit)
        if youngs_modulus is not None:
            material.SetYoungsModulus(youngs_modulus, unit=_modulus_unit(modulus_unit))
        if poisson_ratio is not None:
            material.SetPoissonRatio(poisson_ratio)
        if specific_heat is not None:
            material.SetSpecificHeat(specific_heat, unit="J/kg.K")
        if thermal_conductivity is not None:
            material.SetThermalConductivity(thermal_conductivity, unit="W/m.K")
    except Exception as exc:
        raise ToolError(f"Could not create the material: {exc}") from exc

    assigned = None
    if particle_id is not None:
        particle = _resolve(session, particle_id, "particle")
        try:
            particle.SetMaterial(material)
        except Exception as exc:
            raise ToolError(f"Could not assign the material to {particle_id}: {exc}") from exc
        assigned = particle_id

    assigned_wall = None
    if wall_id is not None:
        wall = _resolve(session, wall_id, "wall")
        try:
            wall.SetMaterial(material)
        except Exception as exc:
            raise ToolError(f"Could not assign the material to {wall_id}: {exc}") from exc
        assigned_wall = wall_id

    assigned_conveyor = None
    if conveyor_id is not None:
        conveyor = _resolve(session, conveyor_id, "conveyor")
        try:
            conveyor.SetMaterial(material)
        except Exception as exc:
            raise ToolError(f"Could not assign the material to {conveyor_id}: {exc}") from exc
        assigned_conveyor = conveyor_id

    ident = session.registry.add(
        "material",
        material,
        label=material.GetName(),
        meta={"density": material.GetDensity("kg/m3")},
    )
    return envelope(
        session,
        message=f"Created material {material.GetName()}.",
        material_id=ident,
        name=material.GetName(),
        density=material.GetDensity("kg/m3"),
        youngs_modulus=material.GetYoungsModulus("N/m2"),
        poisson_ratio=material.GetPoissonRatio(),
        specific_heat_j_kg_k=material.GetSpecificHeat("J/kg.K"),
        thermal_conductivity_w_m_k=material.GetThermalConductivity("W/m.K"),
        particle_id=assigned,
        wall_id=assigned_wall,
        conveyor_id=assigned_conveyor,
    )


async def rocky_set_contact_models(
    ctx: Context,
    normal_force: str | None = None,
    tangential_force: str | None = None,
    rolling_resistance: str | None = None,
    adhesion: str | None = None,
    restitution_model: str | None = None,
    thermal: bool | None = None,
) -> dict[str, Any]:
    """Set the Rocky contact models on the study physics.

    Parameters
    ----------
    normal_force : str | None, default: None
        One of ``linear_hysteresis``, ``linear_elastic_viscous``,
        ``damped_hertzian``, or ``custom``.
    tangential_force : str | None, default: None
        One of ``elastic_coulomb``, ``coulomb_limit``,
        ``mindlin_deresiewicz``, or ``custom``.
    rolling_resistance : str | None, default: None
        One of ``type_1``, ``type_3``, ``none``, or ``custom``.
    adhesion : str | None, default: None
        One of ``none``, ``constant``, ``linear``, ``JKR``, or ``custom``.
    restitution_model : str | None, default: None
        One of ``constant`` or ``velocity_dependent``.
    thermal : bool | None, default: None
        Enable or disable Rocky's thermal model.

    """
    session = get_session(ctx)
    study = require_study(session)
    if normal_force is not None and normal_force not in NORMAL_FORCE_MODELS:
        raise ToolError(
            f"Unknown normal force model {normal_force!r}. Expected one of {NORMAL_FORCE_MODELS}."
        )
    if tangential_force is not None and tangential_force not in TANGENTIAL_FORCE_MODELS:
        raise ToolError(
            f"Unknown tangential force model {tangential_force!r}. "
            f"Expected one of {TANGENTIAL_FORCE_MODELS}."
        )
    if restitution_model is not None and restitution_model not in RESTITUTION_MODELS:
        legal = ", ".join(RESTITUTION_MODELS)
        raise ToolError(
            f"Unknown restitution model {restitution_model!r}. Expected one of {legal}."
        )
    if adhesion is not None and adhesion not in ADHESION_MODELS:
        raise ToolError(f"Unknown adhesion model {adhesion!r}. Expected one of {ADHESION_MODELS}.")
    if rolling_resistance is not None and rolling_resistance not in ROLLING_RESISTANCE_MODELS:
        raise ToolError(
            f"Unknown rolling resistance model {rolling_resistance!r}. "
            f"Expected one of {ROLLING_RESISTANCE_MODELS}."
        )
    try:
        physics = study.GetPhysics()
        if normal_force is not None:
            physics.SetNormalForceModel(normal_force)
        if tangential_force is not None:
            physics.SetTangentialForceModel(tangential_force)
        if rolling_resistance is not None:
            physics.SetRollingResistanceModel(rolling_resistance)
        if adhesion is not None:
            physics.SetAdhesionModel(adhesion)
        if restitution_model is not None:
            physics.SetRestitutionModel(restitution_model)
        if thermal is not None:
            physics.SetEnableThermalModel(thermal)
    except Exception as exc:
        raise ToolError(f"Could not set the contact models: {exc}") from exc
    return envelope(
        session,
        message="Contact models updated.",
        normal_force=physics.GetNormalForceModel(),
        tangential_force=physics.GetTangentialForceModel(),
        rolling_resistance=physics.GetRollingResistanceModel(),
        adhesion=physics.GetAdhesionModel(),
        restitution_model=physics.GetRestitutionModel(),
        thermal=bool(physics.GetEnableThermalModel()),
    )


async def rocky_set_material_interaction(
    ctx: Context,
    material_a: str,
    material_b: str,
    static_friction: float | None = None,
    dynamic_friction: float | None = None,
    restitution: float | None = None,
    adhesive_distance: float | None = None,
    tangential_stiffness_ratio: float | None = None,
    velocity_exponent: float | None = None,
    velocity_limit: float | None = None,
    contact_stiffness_multiplier: float | None = None,
    adhesive_fraction: float | None = None,
    surface_energy: float | None = None,
) -> dict[str, Any]:
    """Set contact coefficients between two solid materials.

    Parameters
    ----------
    material_a : str
        First material id, for example ``material_1``.
    material_b : str
        Second material id. Use the same id twice for a material with itself.
    static_friction : float | None, default: None
        Static friction coefficient.
    dynamic_friction : float | None, default: None
        Dynamic friction coefficient.
    restitution : float | None, default: None
        Restitution coefficient.
    adhesive_distance : float | None, default: None
        Adhesive distance in meters.
    tangential_stiffness_ratio : float | None, default: None
        Tangential stiffness divided by normal stiffness.
    velocity_exponent : float | None, default: None
        Exponent for a velocity-dependent restitution model.
    velocity_limit : float | None, default: None
        Velocity limit for a velocity-dependent restitution model, in m/s.
    contact_stiffness_multiplier : float | None, default: None
        Multiplier on the contact stiffness. Dimensionless.
    adhesive_fraction : float | None, default: None
        Fraction of the contact area that is adhesive. Dimensionless.
    surface_energy : float | None, default: None
        Surface energy of this material pair, in J/m2.

    """
    session = get_session(ctx)
    study = require_study(session)
    first = _resolve(session, material_a, "material")
    second = _resolve(session, material_b, "material")
    try:
        interaction = study.GetMaterialsInteractionCollection().GetMaterialsInteraction(
            first, second
        )
        if static_friction is not None:
            interaction.SetStaticFriction(static_friction)
        if dynamic_friction is not None:
            interaction.SetDynamicFriction(dynamic_friction)
        if restitution is not None:
            interaction.SetRestitutionCoefficient(restitution)
        if adhesive_distance is not None:
            interaction.SetAdhesiveDistance(adhesive_distance, unit="m")
        if tangential_stiffness_ratio is not None:
            interaction.SetTangentialStiffnessRatio(tangential_stiffness_ratio)
        if velocity_exponent is not None:
            interaction.SetVelocityExponent(velocity_exponent)
        if velocity_limit is not None:
            interaction.SetVelocityLimit(velocity_limit, unit="m/s")
        if contact_stiffness_multiplier is not None:
            interaction.SetContactStiffnessMultiplier(contact_stiffness_multiplier)
        if adhesive_fraction is not None:
            interaction.SetAdhesiveFraction(adhesive_fraction)
        if surface_energy is not None:
            interaction.SetSurfaceEnergy(surface_energy, unit="J/m2")
    except Exception as exc:
        raise ToolError(f"Could not set the material interaction: {exc}") from exc
    return envelope(
        session,
        message="Material interaction updated.",
        material_a=material_a,
        material_b=material_b,
        static_friction=interaction.GetStaticFriction(),
        dynamic_friction=interaction.GetDynamicFriction(),
        restitution=interaction.GetRestitutionCoefficient(),
        adhesive_distance_m=interaction.GetAdhesiveDistance("m"),
        tangential_stiffness_ratio=interaction.GetTangentialStiffnessRatio(),
        velocity_exponent=interaction.GetVelocityExponent(),
        velocity_limit_m_s=interaction.GetVelocityLimit("m/s"),
        contact_stiffness_multiplier=interaction.GetContactStiffnessMultiplier(),
        adhesive_fraction=interaction.GetAdhesiveFraction(),
        surface_energy=interaction.GetSurfaceEnergy("J/m2"),
    )


async def rocky_set_temperature(
    ctx: Context,
    object_id: str,
    temperature: float,
) -> dict[str, Any]:
    """Set a temperature in kelvin on a particle inlet, wall, or conveyor.

    Parameters
    ----------
    object_id : str
        Identifier of a particle inlet, wall, or conveyor.
    temperature : float
        Temperature in kelvin. Rocky's token is ``K``.

    """
    session = get_session(ctx)
    require_study(session)
    entry = session.registry.get(object_id)
    if entry is None:
        known = ", ".join(item["id"] for item in session.registry.list()) or "(none)"
        raise ToolError(f"Unknown object id {object_id!r}. Known ids: {known}.")
    try:
        if entry.category == "inlet":
            properties = entry.obj.GetInputPropertiesList()
            if not properties:
                raise ToolError("This inlet has no temperature.")
            properties[0].SetTemperature(temperature, unit="K")
            stored = properties[0].GetTemperature("K")
        elif entry.category in ("wall", "conveyor"):
            entry.obj.SetTemperature(temperature, unit="K")
            stored = entry.obj.GetTemperature("K")
        else:
            raise ToolError(
                f"Object {object_id} is a {entry.category}. "
                "Temperature applies to a particle inlet, wall, or conveyor."
            )
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not set the temperature: {exc}") from exc
    return envelope(
        session,
        message=f"Set the temperature of {object_id} to {stored} K.",
        object_id=object_id,
        category=entry.category,
        temperature_k=stored,
    )


def register(mcp: Any) -> None:
    """Register the setup tools on the MCP server."""
    mcp.tool()(rocky_import_wall)
    mcp.tool()(rocky_create_surface)
    mcp.tool()(rocky_create_feed_conveyor)
    mcp.tool()(rocky_create_receiving_conveyor)
    mcp.tool()(rocky_create_particle)
    mcp.tool()(rocky_set_size_distribution)
    mcp.tool()(rocky_create_material)
    mcp.tool()(rocky_set_material_interaction)
    mcp.tool()(rocky_create_inlet)
    mcp.tool()(rocky_create_volumetric_inlet)
    mcp.tool()(rocky_create_outlet)
    mcp.tool()(rocky_set_domain)
    mcp.tool()(rocky_set_simulation_settings)
    mcp.tool()(rocky_set_gravity)
    mcp.tool()(rocky_set_contact_models)
    mcp.tool()(rocky_set_motion)
    mcp.tool()(rocky_set_temperature)
