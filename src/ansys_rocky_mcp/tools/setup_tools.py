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
BREAKAGE_MODELS = (
    "none",
    "ab_t10",
    "professor_tavares",
    "instantaneous_custom",
    "griffith_surface_energy",
    "shear_stress_criterion",
    "tensile_stress_criterion",
    "tensile_or_shear_criterion",
    "von_mises_stress_criterion",
    "discrete_custom",
)
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
THERMAL_CORRECTION_MODELS = ("none", "morris_et_al_area", "morris_et_al_area_time")
IMPACT_ENERGY_MODELS = ("default", "custom")
SPH_THERMAL_MODELS = ("cleary", "custom")


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


def _angular_acceleration_rad_s2(values: list[float], unit: str) -> tuple[float, float, float]:
    """Return an angular acceleration in rad/s2. Degrees per second squared are converted."""
    if len(values) != 3:
        raise ToolError("angular_acceleration must be three numbers.")
    if unit == "rad/s2":
        scale = 1.0
    elif unit in ("deg/s2", "dega/s2"):
        scale = math.pi / 180.0
    else:
        raise ToolError(
            "Unknown angular acceleration unit "
            f"{unit!r}. Expected 'rad/s2', 'deg/s2', or 'dega/s2'."
        )
    return tuple(float(value) * scale for value in values)  # type: ignore[return-value]


def _angle_rad(value: float, unit: str, label: str) -> float:
    """Return an angle in radians. Degrees are converted."""
    if unit == "rad":
        return float(value)
    if unit in ("deg", "dega"):
        return float(value) * math.pi / 180.0
    raise ToolError(f"Unknown {label} unit {unit!r}. Expected 'rad', 'deg', or 'dega'.")


_TRANSLATION_DEFINITIONS = {
    "fixed_velocity": "fixed_velocity",
    "initial_velocity_and_acceleration": "initial_velocity_and_acceleration",
    "fixed_acceleration": "initial_velocity_and_acceleration",
    "initial_and_final_velocity": "initial_and_final_velocity",
    "initial_and_final_velocities": "initial_and_final_velocity",
}


def _translation_definition(define_as: str | None, has_acceleration: bool, has_final: bool) -> str:
    """Pick the Rocky translation definition so acceleration is not ignored."""
    if define_as is not None:
        token = _TRANSLATION_DEFINITIONS.get(define_as.strip().lower())
        if token is None:
            raise ToolError(
                f"Unknown translation definition {define_as!r}. "
                "Expected 'fixed_velocity', 'initial_velocity_and_acceleration', "
                "or 'initial_and_final_velocity'."
            )
        return token
    if has_acceleration and has_final:
        raise ToolError(
            "Set 'define_as' when both 'linear_acceleration' and 'final_velocity' are given."
        )
    if has_final:
        return "initial_and_final_velocity"
    if has_acceleration:
        return "initial_velocity_and_acceleration"
    return "fixed_velocity"


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


_GEOMETRY_SPLIT_METHODS = ("MODEL", "ASSEMBLY", "PARTS", "BODY")


async def rocky_import_wall(
    ctx: Context,
    path: str,
    scale: float = 1.0,
    name_prefix: str | None = None,
    convert_yz: bool = False,
    geometry_split_method: str = "BODY",
    normal_angle_tolerance: float | None = None,
    max_edge_size: float | None = None,
    triangle_size: float | None = None,
    translation: list[float] | None = None,
    rotation_deg: list[float] | None = None,
    periodic_replication: bool | None = None,
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
    convert_yz : bool, default: False
        Swap Y and Z on import. Use this for geometry saved Z-up.
    geometry_split_method : str, default: "BODY"
        How a CAD file is split: ``MODEL``, ``ASSEMBLY``, ``PARTS``, or ``BODY``.
    normal_angle_tolerance : float | None, default: None
        Maximum mesh angle in degrees for a CAD file.
    max_edge_size : float | None, default: None
        Maximum mesh edge length in meters for a CAD file.
    triangle_size : float | None, default: None
        Wall triangle size in meters, used for contact resolution.
    translation : list[float] | None, default: None
        Move the imported walls by ``[x, y, z]`` in meters.
    rotation_deg : list[float] | None, default: None
        Rotate the imported walls by ``[x, y, z]`` in degrees. Rocky's token is ``dega``.
    periodic_replication : bool | None, default: None
        Copy the wall across a periodic domain boundary.

    """
    session = get_session(ctx)
    study = require_study(session)
    target = Path(path).expanduser()
    if not target.exists():
        raise ToolError(f"Geometry file not found: {target}")
    if geometry_split_method not in _GEOMETRY_SPLIT_METHODS:
        raise ToolError(
            f"Unknown geometry split method {geometry_split_method!r}. "
            f"Expected one of {_GEOMETRY_SPLIT_METHODS}."
        )
    try:
        walls = study.ImportWall(
            str(target),
            import_scale=scale,
            convert_yz=convert_yz,
            custom_name_prefix=name_prefix,
            geometry_split_method=geometry_split_method,
            normal_angle_tolerance=(
                16.0 if normal_angle_tolerance is None else normal_angle_tolerance
            ),
            max_edge_size=max_edge_size,
        )
    except TypeError:
        # Older Rocky builds do not accept every keyword argument.
        walls = study.ImportWall(str(target))
    except Exception as exc:
        raise ToolError(f"Could not import the wall {target}: {exc}") from exc

    walls = list(walls or [])
    if triangle_size is not None:
        for wall in walls:
            wall.SetTriangleSize(triangle_size, unit="m")
    offset = _three_vector(translation, "translation")
    if offset is not None:
        for wall in walls:
            wall.SetTranslation(offset, unit="m")
    angles = _three_vector(rotation_deg, "rotation_deg")
    if angles is not None:
        for wall in walls:
            wall.SetOrientationFromAngles(angles, unit="dega")
    if periodic_replication is not None:
        for wall in walls:
            wall.SetPeriodicReplication(periodic_replication)
    ids = [session.registry.add("wall", wall, label=f"Wall from {target.name}") for wall in walls]
    return envelope(
        session,
        message=f"Imported {len(walls)} wall(s) from {target}.",
        wall_ids=ids,
        convert_yz=bool(walls[0].GetConvertYZ()) if walls else convert_yz,
        geometry_split_method=(
            walls[0].GetGeometrySplitMethod() if walls else geometry_split_method
        ),
        normal_angle_tolerance=(
            walls[0].GetNormalAngleTolerance() if walls else normal_angle_tolerance
        ),
        max_edge_size_m=walls[0].GetMaxEdgeSize() if walls else max_edge_size,
        triangle_size_m=walls[0].GetTriangleSize("m") if walls else triangle_size,
        translation_m=walls[0].GetTranslation("m") if walls else translation,
        rotation_deg=walls[0].GetOrientation("dega") if walls else rotation_deg,
        periodic_replication=(
            bool(walls[0].GetPeriodicReplication()) if walls else periodic_replication
        ),
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
    return_belt_deg: float | None = None,
    out_of_plane_offset: float | None = None,
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
    return_belt_deg : float | None, default: None
        Return-belt angle in degrees. Rocky stores this as ``dega``, not radians.
    out_of_plane_offset : float | None, default: None
        Offset normal to the belt plane, in meters.
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
        return_belt_deg=return_belt_deg,
        out_of_plane_offset=out_of_plane_offset,
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
    return_belt_deg: float | None = None,
    out_of_plane_offset: float | None = None,
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
        if return_belt_deg is not None:
            conveyor.SetReturnBeltAngle(return_belt_deg, unit="dega")
        if out_of_plane_offset is not None:
            conveyor.SetOutOfPlaneOffset(out_of_plane_offset, unit="m")
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
        return_belt_deg=conveyor.GetReturnBeltAngle("dega"),
        out_of_plane_offset_m=conveyor.GetOutOfPlaneOffset("m"),
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
    return_belt_deg: float | None = None,
    out_of_plane_offset: float | None = None,
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
        return_belt_deg=return_belt_deg,
        out_of_plane_offset=out_of_plane_offset,
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
    size_type: str | None = None,
    rolling_resistance: float | None = None,
    breakage_model: str | None = None,
    abt10_reference_size: float | None = None,
    abt10_reference_energy: float | None = None,
    tavares_a: float | None = None,
    tavares_d0: float | None = None,
    tavares_e_inf: float | None = None,
    shear_stress_limit: float | None = None,
    tensile_stress_limit: float | None = None,
    von_mises_stress_limit: float | None = None,
    stress_unit: str = "N/m2",
    plastic_limit: float | None = None,
    porosity: float | None = None,
    bending_angle_deg: float | None = None,
    torsion_angle_deg: float | None = None,
    flexible: bool | None = None,
    elasticity: str | None = None,
    target_element_count: int | None = None,
    elastic_ratio_bending: float | None = None,
    elastic_ratio_normal: float | None = None,
    elastic_ratio_tangential: float | None = None,
    elastic_ratio_torsion: float | None = None,
    element_damping_ratio: float | None = None,
    joint_damping_ratio: float | None = None,
    with_failure: bool | None = None,
    failure_ratio: float | None = None,
    random_angle_deg: list[float] | None = None,
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
    size_type : str | None, default: None
        ``sieve``, ``equivalent_diameter``, or ``original_size_scale``.
    rolling_resistance : float | None, default: None
        Particle rolling resistance coefficient.
    breakage_model : str | None, default: None
        One of Rocky's breakage models. Any value other than ``none`` enables breakage.
    abt10_reference_size : float | None, default: None
        Abt10 reference size in meters.
    abt10_reference_energy : float | None, default: None
        Abt10 reference minimum specific energy in ``J/kg``.
    tavares_a : float | None, default: None
        Tavares A parameter. Rocky's token is ``%``.
    tavares_d0 : float | None, default: None
        Tavares reference size in meters.
    tavares_e_inf : float | None, default: None
        Tavares upper specific energy in ``J/kg``.
    shear_stress_limit : float | None, default: None
        Shear breakage limit. Rocky's token is ``N/m2``. ``Pa`` is accepted.
    tensile_stress_limit : float | None, default: None
        Tensile breakage limit, in the same unit as ``shear_stress_limit``.
    von_mises_stress_limit : float | None, default: None
        Von Mises breakage limit, in the same unit as ``shear_stress_limit``.
    stress_unit : str, default: "N/m2"
        Unit of the stress limits. ``Pa`` is sent to Rocky as ``N/m2``.
    plastic_limit : float | None, default: None
        Plastic limit in pascals. Rocky's token for this one is ``Pa``.
    porosity : float | None, default: None
        Particle porosity. Rocky's token is ``<fraction>``.
    bending_angle_deg : float | None, default: None
        Bending angle limit in degrees. Rocky's token is ``dega``.
    torsion_angle_deg : float | None, default: None
        Torsion angle limit in degrees. Rocky's token is ``dega``.
    flexible : bool | None, default: None
        Make the particle flexible. A bending or torsion limit turns this on.
    elasticity : str | None, default: None
        ``isotropic`` or ``anisotropic``.
    target_element_count : int | None, default: None
        Number of elements in a flexible fiber. Must be at least 1.
    elastic_ratio_bending : float | None, default: None
        Bending stiffness relative to the base stiffness.
    elastic_ratio_normal : float | None, default: None
        Normal stiffness relative to the base stiffness.
    elastic_ratio_tangential : float | None, default: None
        Tangential stiffness relative to the base stiffness.
    elastic_ratio_torsion : float | None, default: None
        Torsion stiffness relative to the base stiffness.
    element_damping_ratio : float | None, default: None
        Damping inside each fiber element.
    joint_damping_ratio : float | None, default: None
        Damping at the joints between fiber elements.
    with_failure : bool | None, default: None
        Allow a flexible particle to break. A failure ratio turns this on.
    failure_ratio : float | None, default: None
        Strain ratio at which a flexible particle breaks.
    random_angle_deg : list[float] | None, default: None
        Half-range of a random orientation, ``[x, y, z]`` in degrees.
        Rocky stores this as ``dega``. Setting it turns random orientation on.
    name : str | None, default: None
        Name for the created particle.

    """
    if shape not in PARTICLE_SHAPES:
        raise ToolError(f"Unknown particle shape {shape!r}. Expected one of {PARTICLE_SHAPES}.")
    valid_sizes = ("sieve", "equivalent_diameter", "original_size_scale")
    if size_type is not None and size_type not in valid_sizes:
        raise ToolError(f"Unknown size type {size_type!r}. Expected one of {valid_sizes}.")
    if elasticity is not None and elasticity not in ("isotropic", "anisotropic"):
        raise ToolError("elasticity must be 'isotropic' or 'anisotropic'.")
    if target_element_count is not None and target_element_count < 1:
        raise ToolError("target_element_count must be at least 1.")
    if breakage_model is not None and breakage_model not in BREAKAGE_MODELS:
        raise ToolError(
            f"Unknown breakage model {breakage_model!r}. Expected one of {BREAKAGE_MODELS}."
        )
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
        if size_type is not None:
            particle.SetSizeType(size_type)
        if rolling_resistance is not None:
            particle.SetRollingResistance(rolling_resistance)
        if breakage_model is not None:
            particle.SetBreakageModel(breakage_model)
        if abt10_reference_size is not None:
            particle.SetAbt10ReferenceSize(abt10_reference_size, unit="m")
        if abt10_reference_energy is not None:
            particle.SetAbt10ReferenceMinimumSpecificEnergy(abt10_reference_energy, unit="J/kg")
        if tavares_a is not None:
            particle.SetTavaresA(tavares_a, unit="%")
        if tavares_d0 is not None:
            particle.SetTavaresD0(tavares_d0, unit="m")
        if tavares_e_inf is not None:
            particle.SetTavaresEInf(tavares_e_inf, unit="J/kg")
        if any(
            value is not None
            for value in (shear_stress_limit, tensile_stress_limit, von_mises_stress_limit)
        ):
            stress = _modulus_unit(stress_unit)
            if shear_stress_limit is not None:
                particle.SetShearStressLimit(shear_stress_limit, unit=stress)
            if tensile_stress_limit is not None:
                particle.SetTensileStressLimit(tensile_stress_limit, unit=stress)
            if von_mises_stress_limit is not None:
                particle.SetVonMisesStressLimit(von_mises_stress_limit, unit=stress)
        if plastic_limit is not None:
            particle.SetPlasticLimit(plastic_limit, unit="Pa")
        if porosity is not None:
            particle.SetPorosity(porosity, unit="<fraction>")
        if bending_angle_deg is not None:
            particle.SetBendingAngleLimit(bending_angle_deg, unit="dega")
        if torsion_angle_deg is not None:
            particle.SetTorsionAngleLimit(torsion_angle_deg, unit="dega")
        if flexible is None and (
            bending_angle_deg is not None
            or torsion_angle_deg is not None
            or elasticity is not None
            or target_element_count is not None
            or elastic_ratio_bending is not None
            or elastic_ratio_normal is not None
            or elastic_ratio_tangential is not None
            or elastic_ratio_torsion is not None
            or element_damping_ratio is not None
            or joint_damping_ratio is not None
            or with_failure is not None
            or failure_ratio is not None
        ):
            flexible = True
        if flexible is not None:
            particle.SetFlexible(flexible)
        if elasticity is not None:
            particle.SetElasticity(elasticity)
        if target_element_count is not None:
            particle.SetTargetNumberOfElements(target_element_count)
        if elastic_ratio_bending is not None:
            particle.SetElasticRatioBending(elastic_ratio_bending)
        if elastic_ratio_normal is not None:
            particle.SetElasticRatioNormal(elastic_ratio_normal)
        if elastic_ratio_tangential is not None:
            particle.SetElasticRatioTangential(elastic_ratio_tangential)
        if elastic_ratio_torsion is not None:
            particle.SetElasticRatioTorsion(elastic_ratio_torsion)
        if element_damping_ratio is not None:
            particle.SetElementDampingRatio(element_damping_ratio)
        if joint_damping_ratio is not None:
            particle.SetJointDampingRatio(joint_damping_ratio)
        if failure_ratio is not None and with_failure is None:
            with_failure = True
        if with_failure is not None:
            particle.SetWithFailure(with_failure)
        if failure_ratio is not None:
            particle.SetFailureRatio(failure_ratio)
        if random_angle_deg is not None:
            particle.SetRandomAnglesHalfRange(
                _three_vector(random_angle_deg, "random_angle_deg"),
                unit="dega",
            )
            particle.SetEnableRandomAngle(True)
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
        size_type=particle.GetSizeType(),
        rolling_resistance=particle.GetRollingResistance(),
        breakage_model=particle.GetBreakageModel(),
        breakage_enabled=bool(particle.GetEnableBreakage()),
        abt10_reference_size_m=particle.GetAbt10ReferenceSize("m"),
        abt10_reference_energy_j_kg=particle.GetAbt10ReferenceMinimumSpecificEnergy("J/kg"),
        tavares_a_percent=particle.GetTavaresA("%"),
        tavares_d0_m=particle.GetTavaresD0("m"),
        tavares_e_inf_j_kg=particle.GetTavaresEInf("J/kg"),
        shear_stress_limit_n_m2=particle.GetShearStressLimit("N/m2"),
        tensile_stress_limit_n_m2=particle.GetTensileStressLimit("N/m2"),
        von_mises_stress_limit_n_m2=particle.GetVonMisesStressLimit("N/m2"),
        plastic_limit_pa=particle.GetPlasticLimit("Pa"),
        porosity=particle.GetPorosity("<fraction>"),
        bending_angle_deg=particle.GetBendingAngleLimit("dega"),
        torsion_angle_deg=particle.GetTorsionAngleLimit("dega"),
        flexible=bool(particle.GetFlexible()),
        elasticity=particle.GetElasticity(),
        target_element_count=particle.GetTargetNumberOfElements(),
        elastic_ratio_bending=particle.GetElasticRatioBending(),
        elastic_ratio_normal=particle.GetElasticRatioNormal(),
        elastic_ratio_tangential=particle.GetElasticRatioTangential(),
        elastic_ratio_torsion=particle.GetElasticRatioTorsion(),
        element_damping_ratio=particle.GetElementDampingRatio(),
        joint_damping_ratio=particle.GetJointDampingRatio(),
        with_failure=bool(particle.GetWithFailure()),
        failure_ratio=particle.GetFailureRatio(),
        random_angle=bool(particle.GetEnableRandomAngle()),
        random_angle_deg=particle.GetRandomAnglesHalfRange("dega"),
    )


async def rocky_set_size_distribution(
    ctx: Context,
    particle_id: str,
    model: str,
    minimum_size: float | None = None,
    minimum_size_ratio: float | None = None,
    surface_energy: float | None = None,
    unit: str = "m",
    energy_unit: str = "J/m2",
    sizes: list[float] | None = None,
    cumulative_percent: list[float] | None = None,
    size_type: str | None = None,
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
    minimum_size_ratio : float | None, default: None
        Minimum size as a fraction of the maximum size.
    unit : str, default: "m"
        Unit of ``minimum_size``.
    surface_energy : float | None, default: None
        Surface energy used by an adhesion model such as JKR.
    energy_unit : str, default: "J/m2"
        Unit of ``surface_energy``.
    sizes : list[float] | None, default: None
        Sieve sizes for a custom table. Rocky's default token is ``m``.
    cumulative_percent : list[float] | None, default: None
        Cumulative percent for each size. Rocky stores a plain number.
    size_type : str | None, default: None
        One of ``sieve``, ``equivalent_diameter``, or ``original_size_scale``.

    """
    session = get_session(ctx)
    require_study(session)
    if model not in SIZE_DISTRIBUTIONS:
        raise ToolError(
            f"Unknown size distribution {model!r}. Expected one of {SIZE_DISTRIBUTIONS}."
        )
    if (sizes is None) != (cumulative_percent is None):
        raise ToolError("'sizes' and 'cumulative_percent' must be given together.")
    if sizes is not None and len(sizes) != len(cumulative_percent or []):
        raise ToolError("'sizes' and 'cumulative_percent' must have the same length.")
    size_types = ("sieve", "equivalent_diameter", "original_size_scale")
    if size_type is not None and size_type not in size_types:
        raise ToolError(f"Unknown size type {size_type!r}. Expected one of {size_types}.")
    particle = _resolve(session, particle_id, "particle")
    try:
        particle.SetDistributionModel(model)
        if minimum_size is not None:
            particle.SetMinimumSize(minimum_size, unit=unit)
        if minimum_size_ratio is not None:
            particle.SetMinimumSizeRatio(minimum_size_ratio)
        if surface_energy is not None:
            particle.SetSurfaceEnergy(surface_energy, unit=energy_unit)
        if sizes is not None or size_type is not None:
            listing = particle.GetSizeDistributionList()
            if size_type is not None:
                listing.SetSizeType(size_type)
            if sizes is not None:
                listing.Clear()
                for size, percent in zip(sizes, cumulative_percent or [], strict=True):
                    entry = listing.New()
                    entry.SetSize(size, unit)
                    entry.SetCumulativePercentage(percent)
    except Exception as exc:
        raise ToolError(f"Could not set the size distribution: {exc}") from exc
    return envelope(
        session,
        message="Size distribution updated.",
        particle_id=particle_id,
        model=particle.GetDistributionModel(),
        minimum_size_m=particle.GetMinimumSize("m"),
        minimum_size_ratio=particle.GetMinimumSizeRatio(),
        surface_energy=particle.GetSurfaceEnergy("J/m2"),
        sizes_m=(
            [entry.GetSize("m") for entry in particle.GetSizeDistributionList()]
            if sizes is not None
            else None
        ),
        cumulative_percent=(
            [entry.GetCumulativePercentage() for entry in particle.GetSizeDistributionList()]
            if sizes is not None
            else None
        ),
        size_type=particle.GetSizeType() if size_type is not None else None,
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
    start_time: float | None = None,
    stop_time: float | None = None,
    target_velocity: float | None = None,
    periodic: bool | None = None,
    period: float | None = None,
    velocity_x: float | None = None,
    velocity_z: float | None = None,
    stop_all: bool | None = None,
    force_packing: bool | None = None,
    sph_temperature: float | None = None,
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
    start_time : float | None, default: None
        Time in seconds when the inlet starts. Rocky's token is ``s``.
    stop_time : float | None, default: None
        Time in seconds when the inlet stops.
    target_velocity : float | None, default: None
        Injection speed normal to the inlet, in meters per second.
    periodic : bool | None, default: None
        Repeat the inlet on a period. A ``period`` turns this on.
    period : float | None, default: None
        Repeat period in seconds. Rocky's token is ``s``.
    velocity_x : float | None, default: None
        Local x velocity in meters per second.
    velocity_z : float | None, default: None
        Local z velocity in meters per second.
    stop_all : bool | None, default: None
        Stop every particle at ``stop_time``. A stop time turns this on.
    force_packing : bool | None, default: None
        Pack particles into the inlet even when they overlap.
    sph_temperature : float | None, default: None
        SPH fluid temperature in kelvin. Setting it turns SPH injection on.

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
        if start_time is not None:
            inlet.SetStartTime(start_time, unit="s")
        if stop_time is not None:
            inlet.SetStopTime(stop_time, unit="s")
            if stop_all is None:
                stop_all = True
        if stop_all is not None:
            inlet.SetStopAllAtStopTime(stop_all)
        if force_packing is not None:
            inlet.SetForcePacking(force_packing)
        if sph_temperature is not None:
            inlet.SetSphTemperature(sph_temperature, unit="K")
            inlet.SetSphInjectionEnabled(True)
        if target_velocity is not None:
            inlet.SetTargetNormalVelocity(target_velocity, unit="m/s")
        if velocity_x is not None:
            inlet.SetUxLocal(velocity_x, unit="m/s")
        if velocity_z is not None:
            inlet.SetUzLocal(velocity_z, unit="m/s")
        if any(value is not None for value in (target_velocity, velocity_x, velocity_z)):
            inlet.SetUseTargetNormalVelocity(True)
        if period is not None and periodic is None:
            periodic = True
        if periodic is not None:
            inlet.SetPeriodic(periodic)
        if period is not None:
            inlet.SetPeriod(period, unit="s")
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
        start_time_s=inlet.GetStartTime("s"),
        stop_time_s=inlet.GetStopTime("s"),
        target_velocity_m_s=inlet.GetTargetNormalVelocity("m/s"),
        periodic=bool(inlet.GetPeriodic()),
        period_s=inlet.GetPeriod("s"),
        velocity_x_m_s=inlet.GetUxLocal("m/s"),
        velocity_z_m_s=inlet.GetUzLocal("m/s"),
        stop_all=bool(inlet.GetStopAllAtStopTime()),
        force_packing=bool(inlet.GetForcePacking()),
        sph_temperature_k=inlet.GetSphTemperature("K"),
        sph_injection=bool(inlet.GetSphInjectionEnabled()),
    )


def _three_vector(values: list[float] | None, label: str) -> list[float] | None:
    """Require a 3-number sequence, or accept an omitted value."""
    if values is None:
        return None
    if len(values) != 3:
        raise ToolError(f"{label} must be three numbers.")
    return [float(value) for value in values]


async def rocky_add_inlet_particle(
    ctx: Context,
    inlet_id: str,
    particle_id: str,
    mass_flow_rate: float,
    unit: str = "t/h",
) -> dict[str, Any]:
    """Add another particle to an existing particle inlet.

    Parameters
    ----------
    inlet_id : str
        Identifier of the particle inlet.
    particle_id : str
        Identifier of the particle definition to add.
    mass_flow_rate : float
        Mass flow rate of this particle.
    unit : str, default: "t/h"
        Unit of the mass flow rate. Rocky's default token is ``t/h``.

    """
    session = get_session(ctx)
    require_study(session)
    inlet = _resolve(session, inlet_id, "inlet")
    particle = _resolve(session, particle_id, "particle")
    try:
        inlet.AddParticle(particle)
        chosen = None
        for properties in inlet.GetInputPropertiesList():
            if properties.GetParticle() is particle:
                chosen = properties
        if chosen is None:
            raise ToolError("The inlet did not record the added particle.")
        chosen.SetMassFlowRate(mass_flow_rate, unit=unit)
        count = len(inlet.GetInputPropertiesList())
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not add the particle to the inlet: {exc}") from exc
    return envelope(
        session,
        message=f"Added {particle_id} to {inlet_id}.",
        inlet_id=inlet_id,
        particle_id=particle_id,
        mass_flow_rate=mass_flow_rate,
        unit=unit,
        particle_count=count,
    )


async def rocky_remove_inlet_particle(
    ctx: Context,
    inlet_id: str,
    particle_id: str,
) -> dict[str, Any]:
    """Remove a particle from an existing particle inlet.

    Parameters
    ----------
    inlet_id : str
        Identifier of the particle inlet.
    particle_id : str
        Identifier of the particle definition to remove.

    """
    session = get_session(ctx)
    require_study(session)
    inlet = _resolve(session, inlet_id, "inlet")
    particle = _resolve(session, particle_id, "particle")
    try:
        inlet.RemoveParticle(particle)
        count = len(inlet.GetInputPropertiesList())
    except Exception as exc:
        raise ToolError(f"Could not remove the particle from the inlet: {exc}") from exc
    return envelope(
        session,
        message=f"Removed {particle_id} from {inlet_id}.",
        inlet_id=inlet_id,
        particle_id=particle_id,
        particle_count=count,
    )


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


def _call_local(solver: Any, method_name: str, *args: Any) -> None:
    """Write a GPU setting onto local execution settings when Rocky exposes it."""
    getter = getattr(solver, "GetLocalExecutionSettings", None)
    if getter is None:
        return
    method = getattr(getter(), method_name, None)
    if method is not None:
        method(*args)


def _local_value(solver: Any, method_name: str) -> Any:
    """Read a value from local execution settings when Rocky exposes it."""
    getter = getattr(solver, "GetLocalExecutionSettings", None)
    if getter is None:
        return None
    method = getattr(getter(), method_name, None)
    if method is None:
        return None
    return method()


def _settings_processors(solver: Any, getter_name: str) -> int | None:
    """Read a processor count from local or MPI settings when Rocky exposes it."""
    getter = getattr(solver, getter_name, None)
    if getter is None:
        return None
    reader = getattr(getter(), "GetNumberOfProcessors", None)
    if reader is None:
        return None
    return int(reader())


async def rocky_set_simulation_settings(
    ctx: Context,
    duration: float | None = None,
    time_interval: float | None = None,
    processors: int | None = None,
    execution_mode: str | None = None,
    curves_frequency: int | None = None,
    timestep_model: str | None = None,
    fixed_timestep: float | None = None,
    processing_unit: str | None = None,
    target_gpus: list[int] | None = None,
    gpu_slicing_direction: str | None = None,
    release_without_overlap_check: bool | None = None,
    breakage_delay: float | None = None,
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
    timestep_model : str | None, default: None
        ``constant`` or ``variable``.
    fixed_timestep : float | None, default: None
        Fixed DEM timestep in seconds. Setting it selects the constant model.
    processing_unit : str | None, default: None
        ``CPU`` or ``GPU``. Rocky's object lists the accepted names.
    target_gpus : list[int] | None, default: None
        GPU ids for a local GPU run, for example ``[0]``.
    gpu_slicing_direction : str | None, default: None
        How a multi-GPU run splits the domain. Rocky lists the accepted names.
    release_without_overlap_check : bool | None, default: None
        Release inlet particles even when they overlap. Needed with force packing.
    breakage_delay : float | None, default: None
        Seconds after release before breakage can start. Rocky's token is ``s``.
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
    if timestep_model is not None and timestep_model not in ("constant", "variable"):
        raise ToolError("timestep_model must be 'constant' or 'variable'.")
    if fixed_timestep is not None and timestep_model is None:
        timestep_model = "constant"
    solver = study.GetSolver()
    try:
        if duration is not None:
            solver.SetSimulationDuration(duration, unit=unit)
        if time_interval is not None:
            solver.SetTimeInterval(time_interval, unit=unit)
        if processors is not None:
            if processors < 1:
                raise ToolError("processors must be at least 1.")
            count = int(processors)
            solver.SetNumberOfProcessors(count)
            for settings_name in ("GetLocalExecutionSettings", "GetMpiSettings"):
                settings_getter = getattr(solver, settings_name, None)
                if settings_getter is None:
                    continue
                settings = settings_getter()
                settings_setter = getattr(settings, "SetNumberOfProcessors", None)
                if settings_setter is not None:
                    settings_setter(count)
        if execution_mode is not None:
            solver.SetExecutionMode(execution_mode)
        if curves_frequency is not None:
            solver.SetSolverCurvesFrequency(int(curves_frequency))
        if timestep_model is not None:
            solver.SetTimestepModel(timestep_model)
        if fixed_timestep is not None:
            solver.SetFixedTimestep(fixed_timestep, unit="s")
            solver.SetUseFixedTimestep(True)
        if processing_unit is not None:
            valid_units = list(solver.GetValidProcessingUnitValues())
            if processing_unit not in valid_units:
                raise ToolError(
                    f"Unknown processing unit {processing_unit!r}. Expected one of {valid_units}."
                )
            solver.SetProcessingUnit(processing_unit)
            _call_local(solver, "SetProcessingUnit", processing_unit)
        if target_gpus is not None:
            solver.SetTargetGpus(target_gpus)
            _call_local(solver, "SetTargetGpus", target_gpus)
        if gpu_slicing_direction is not None:
            valid_directions = list(solver.GetValidMultiGpuSlicingDirectionValues())
            if gpu_slicing_direction not in valid_directions:
                raise ToolError(
                    f"Unknown GPU slicing direction {gpu_slicing_direction!r}. "
                    f"Expected one of {valid_directions}."
                )
            solver.SetMultiGpuSlicingDirection(gpu_slicing_direction)
            _call_local(solver, "SetMultiGpuSlicingDirection", gpu_slicing_direction)
        if release_without_overlap_check is not None:
            solver.SetReleaseParticlesWithoutOverlapCheck(release_without_overlap_check)
        if breakage_delay is not None:
            solver.SetBreakageDelayAfterRelease(breakage_delay, unit="s")
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not configure the solver: {exc}") from exc

    return envelope(
        session,
        message="Solver settings updated.",
        duration_s=solver.GetSimulationDuration("s"),
        time_interval_s=solver.GetTimeInterval("s"),
        processors=solver.GetNumberOfProcessors(),
        local_processors=_settings_processors(solver, "GetLocalExecutionSettings"),
        mpi_processors=_settings_processors(solver, "GetMpiSettings"),
        execution_mode=solver.GetExecutionMode(),
        curves_frequency=solver.GetSolverCurvesFrequency(),
        timestep_model=solver.GetTimestepModel(),
        fixed_timestep_s=solver.GetFixedTimestep("s"),
        use_fixed_timestep=bool(solver.GetUseFixedTimestep()),
        processing_unit=solver.GetProcessingUnit(),
        local_processing_unit=_local_value(solver, "GetProcessingUnit"),
        target_gpus=list(solver.GetTargetGpus()),
        gpu_slicing_direction=solver.GetMultiGpuSlicingDirection(),
        release_without_overlap_check=bool(solver.GetReleaseParticlesWithoutOverlapCheck()),
        breakage_delay_s=solver.GetBreakageDelayAfterRelease("s"),
    )


_SPH_BOUNDARIES = ("free_slip", "no_slip_laminar", "no_slip_turbulent")


async def rocky_set_wall_mass(
    ctx: Context,
    wall_id: str,
    mass: float,
    inertia: list[float] | None = None,
    disable_time: float | None = None,
) -> dict[str, Any]:
    """Set the mass of a moving wall.

    Parameters
    ----------
    wall_id : str
        Identifier of the wall.
    mass : float
        Boundary mass in kilograms.
    inertia : list[float] | None, default: None
        Principal moments of inertia ``[ix, iy, iz]``. Rocky's token is ``kg.m2``.
    disable_time : float | None, default: None
        Time in seconds when the wall stops interacting. Rocky's token is ``s``.

    """
    session = get_session(ctx)
    require_study(session)
    wall = _resolve(session, wall_id, "wall")
    moments = _three_vector(inertia, "inertia")
    try:
        wall.SetBoundaryMass(mass, unit="kg")
        if moments is not None:
            wall.SetPrincipalMomentOfInertia(moments, unit="kg.m2")
        if disable_time is not None:
            wall.SetDisableTime(disable_time, unit="s")
    except Exception as exc:
        raise ToolError(f"Could not set the wall mass: {exc}") from exc
    return envelope(
        session,
        message=f"Set the mass of {wall_id} to {mass} kg.",
        wall_id=wall_id,
        mass_kg=wall.GetBoundaryMass("kg"),
        inertia_kg_m2=wall.GetPrincipalMomentOfInertia("kg.m2"),
        disable_time_s=wall.GetDisableTime("s"),
    )


async def rocky_set_wall_wear(
    ctx: Context,
    wall_id: str,
    wear_model: str | None = None,
    volume_shear_work_ratio: float | None = None,
) -> dict[str, Any]:
    """Set wall wear. The Archard ratio uses Rocky's ``m3/J`` token.

    Parameters
    ----------
    wall_id : str
        Identifier of the wall.
    wear_model : str | None, default: None
        ``none``, ``archard``, or ``custom``. A ratio selects ``archard``.
    volume_shear_work_ratio : float | None, default: None
        Removed volume per unit shear work, in cubic meters per joule.

    """
    if volume_shear_work_ratio is not None and wear_model is None:
        wear_model = "archard"
    if wear_model is not None and wear_model not in ("none", "archard", "custom"):
        raise ToolError("wear_model must be 'none', 'archard', or 'custom'.")
    session = get_session(ctx)
    require_study(session)
    wall = _resolve(session, wall_id, "wall")
    try:
        if wear_model is not None:
            wall.SetWearModel(wear_model)
        if volume_shear_work_ratio is not None:
            wall.SetVolumeShearWorkRatio(volume_shear_work_ratio, unit="m3/J")
    except Exception as exc:
        raise ToolError(f"Could not set the wall wear: {exc}") from exc
    return envelope(
        session,
        message=f"Set wear on {wall_id}.",
        wall_id=wall_id,
        wear_model=wall.GetWearModel(),
        volume_shear_work_ratio_m3_j=wall.GetVolumeShearWorkRatio("m3/J"),
    )


async def rocky_set_sph_boundary(
    ctx: Context,
    wall_id: str,
    boundary_type: str,
    contact_angle_deg: float | None = None,
    capillary_friction: float | None = None,
) -> dict[str, Any]:
    """Set how an SPH fluid meets a wall.

    Parameters
    ----------
    wall_id : str
        Identifier of the wall.
    boundary_type : str
        ``free_slip``, ``no_slip_laminar``, or ``no_slip_turbulent``.
    contact_angle_deg : float | None, default: None
        Surface-tension contact angle in degrees. Rocky's token is ``dega``.
    capillary_friction : float | None, default: None
        Friction coefficient for the capillary force at the wall.

    """
    if boundary_type not in _SPH_BOUNDARIES:
        raise ToolError(
            f"Unknown SPH boundary type {boundary_type!r}. Expected one of {_SPH_BOUNDARIES}."
        )
    session = get_session(ctx)
    require_study(session)
    wall = _resolve(session, wall_id, "wall")
    try:
        wall.SetSphBoundaryType(boundary_type)
        if contact_angle_deg is not None:
            wall.SetSurfaceTensionContactAngle(contact_angle_deg, unit="dega")
        if capillary_friction is not None:
            wall.SetCapillaryFrictionCoefficient(capillary_friction)
    except Exception as exc:
        raise ToolError(f"Could not set the SPH boundary: {exc}") from exc
    return envelope(
        session,
        message=f"Set the SPH boundary of {wall_id} to {boundary_type}.",
        wall_id=wall_id,
        boundary_type=wall.GetSphBoundaryType(),
        contact_angle_deg=wall.GetSurfaceTensionContactAngle("dega"),
        capillary_friction=wall.GetCapillaryFrictionCoefficient(),
    )


async def rocky_set_output_property(
    ctx: Context,
    names: list[str] | None = None,
    enabled: bool = True,
    modules: bool = False,
) -> dict[str, Any]:
    """Enable a Rocky output so a later run records it.

    Parameters
    ----------
    names : list[str] | None, default: None
        Property path, for example ``["Particles", "Count"]``. Omit it to
        list the names Rocky accepts without changing anything.
    enabled : bool, default: True
        Whether the property is recorded.
    modules : bool, default: False
        When true, the path is a module output rather than a standard output.

    """
    session = get_session(ctx)
    study = require_study(session)
    solver = study.GetSolver()
    if not names:
        return envelope(
            session,
            message="Listed the output properties Rocky can record.",
            available_standard=solver.GetAvailableStandardOutputProperties(),
            available_modules=solver.GetAvailableModulesOutputProperties(),
        )
    try:
        if modules:
            solver.SetModulesOutputPropertyEnabled(*names, enabled=enabled)
            recorded = bool(solver.GetModulesOutputPropertyEnabled(*names))
            available = solver.GetAvailableModulesOutputProperties()
        else:
            solver.SetStandardOutputPropertyEnabled(*names, enabled=enabled)
            recorded = bool(solver.GetStandardOutputPropertyEnabled(*names))
            available = solver.GetAvailableStandardOutputProperties()
    except Exception as exc:
        raise ToolError(f"Could not set the output property: {exc}") from exc
    return envelope(
        session,
        message=f"{'Enabled' if recorded else 'Disabled'} output {'/'.join(names)}.",
        names=list(names),
        enabled=recorded,
        modules=modules,
        available=available,
    )


async def rocky_set_motion(
    ctx: Context,
    kind: str,
    targets: list[str],
    angular_velocity: list[float] | None = None,
    angular_acceleration: list[float] | None = None,
    acceleration_unit: str = "rad/s2",
    linear_velocity: list[float] | None = None,
    linear_acceleration: list[float] | None = None,
    final_velocity: list[float] | None = None,
    define_as: str | None = None,
    origin: list[float] | None = None,
    orientation_deg: list[float] | None = None,
    unit: str = "rad/s",
    start_time: float = 0.0,
    stop_time: float = 1000.0,
    period: float | None = None,
    period_start: float | None = None,
    period_stop: float | None = None,
    periodic: bool | None = None,
    linear_limits: bool | None = None,
    angular_limits: bool | None = None,
    min_linear_m: list[float] | None = None,
    max_linear_m: list[float] | None = None,
    min_angular_deg: list[float] | None = None,
    max_angular_deg: list[float] | None = None,
    frequency_hz: float | None = None,
    amplitude: float | None = None,
    direction: list[float] | None = None,
    phase_deg: float | None = None,
    frequency_variation_hz_s: float = 0.0,
    amplitude_variation: float | None = None,
) -> dict[str, Any]:
    """Apply a rotation, translation, vibration, or pendulum to study objects.

    Parameters
    ----------
    kind : str
        ``"rotation"``, ``"translation"``, ``"vibration"``, or ``"pendulum"``.
    targets : list[str]
        Identifiers of the objects the motion is applied to, for example
        ``["surface_1"]``.
    angular_velocity : list[float] | None, default: None
        Angular velocity vector as ``[wx, wy, wz]``. Required for a rotation.
    angular_acceleration : list[float] | None, default: None
        Angular acceleration ``[ax, ay, az]``. ``deg/s2`` and ``dega/s2`` are
        converted to ``rad/s2``.
    acceleration_unit : str, default: "rad/s2"
        Unit of ``angular_acceleration``.
    linear_velocity : list[float] | None, default: None
        Linear velocity vector as ``[vx, vy, vz]``. Required for a translation.
    linear_acceleration : list[float] | None, default: None
        Linear acceleration ``[ax, ay, az]`` in meters per second squared.
        When this is set and ``define_as`` is omitted, the motion is
        ``initial_velocity_and_acceleration`` so Rocky uses the acceleration.
    final_velocity : list[float] | None, default: None
        Final velocity ``[vx, vy, vz]`` in meters per second. When this is set
        and ``define_as`` is omitted, the motion is ``initial_and_final_velocity``.
    define_as : str | None, default: None
        How Rocky defines the translation: ``fixed_velocity``,
        ``initial_velocity_and_acceleration``, or ``initial_and_final_velocity``.
    origin : list[float] | None, default: None
        Frame position ``[x, y, z]`` in meters. This is the point the rotation
        is taken about. Rocky's token is ``m``.
    orientation_deg : list[float] | None, default: None
        Frame angles ``[x, y, z]`` in degrees. Rocky's token is ``dega``.
    unit : str, default: "rad/s"
        Unit of the angular velocity. Rocky stores ``rad/s``. ``deg/s`` and
        ``dega/s`` are converted before the call. Translations always use ``m/s``.
    start_time : float, default: 0.0
        Start time of the motion in seconds.
    stop_time : float, default: 1000.0
        Stop time of the motion in seconds.
    period : float | None, default: None
        Repeat period in seconds. Rocky's token is ``s``.
    period_start : float | None, default: None
        Time in seconds when the repeating motion starts.
    period_stop : float | None, default: None
        Time in seconds when the repeating motion stops.
    periodic : bool | None, default: None
        Turn Rocky's periodic motion on or off. A period or a repeat window
        turns it on when this is omitted, so Rocky does not ignore the times.
    linear_limits : bool | None, default: None
        Enable free-body travel limits. A linear limit turns this on when omitted.
    angular_limits : bool | None, default: None
        Enable free-body angle limits. An angular limit turns this on when omitted.
    min_linear_m : list[float] | None, default: None
        Minimum free-body travel ``[x, y, z]`` in meters.
    max_linear_m : list[float] | None, default: None
        Maximum free-body travel ``[x, y, z]`` in meters.
    min_angular_deg : list[float] | None, default: None
        Minimum free-body angles ``[x, y, z]``. Rocky's token is ``dega``.
    max_angular_deg : list[float] | None, default: None
        Maximum free-body angles ``[x, y, z]``. Rocky's token is ``dega``.
    frequency_hz : float | None, default: None
        Frequency in hertz. Required for a vibration or a pendulum.
    amplitude : float | None, default: None
        Vibration amplitude in meters, or pendulum amplitude in degrees.
        A pendulum amplitude is converted to radians before Rocky stores it.
    direction : list[float] | None, default: None
        Direction ``[x, y, z]``. Rocky stores it with the length token ``m``.
    phase_deg : float | None, default: None
        Pendulum phase in degrees. Rocky's token is ``rad``.
    frequency_variation_hz_s : float, default: 0.0
        Change of frequency in hertz per second.
    amplitude_variation : float | None, default: None
        Vibration amplitude change in meters per second, or pendulum amplitude
        change in degrees per second. The pendulum value is converted to ``rad/s``.

    """
    session = get_session(ctx)
    study = require_study(session)
    kind = kind.strip().lower()
    if kind not in ("rotation", "translation", "vibration", "pendulum"):
        raise ToolError(
            f"Unknown motion kind {kind!r}. "
            "Expected 'rotation', 'translation', 'vibration', or 'pendulum'."
        )
    if not targets:
        raise ToolError("At least one target object id is required.")

    try:
        source = study.GetMotionFrameSource()
        frame = source.NewFrame()
    except Exception as exc:
        raise ToolError(f"Could not create a motion frame: {exc}") from exc

    radians: tuple[float, float, float] | None = None
    acceleration: tuple[float, float, float] | None = None
    placed = _three_vector(origin, "origin")
    angles = _three_vector(orientation_deg, "orientation_deg")
    axis = _three_vector(direction, "direction")
    min_linear = _three_vector(min_linear_m, "min_linear_m")
    max_linear = _three_vector(max_linear_m, "max_linear_m")
    min_angular = _three_vector(min_angular_deg, "min_angular_deg")
    max_angular = _three_vector(max_angular_deg, "max_angular_deg")
    repeat = periodic
    if repeat is None and (
        period is not None or period_start is not None or period_stop is not None
    ):
        repeat = True
    use_linear = linear_limits
    if use_linear is None and (min_linear is not None or max_linear is not None):
        use_linear = True
    use_angular = angular_limits
    if use_angular is None and (min_angular is not None or max_angular is not None):
        use_angular = True
    amplitude_rad: float | None = None
    phase_rad: float | None = None
    amplitude_rate_rad_s: float | None = None
    try:
        if placed is not None:
            frame.SetRelativePosition(placed, unit="m")
        if angles is not None:
            frame.SetOrientationFromAngles(tuple(angles), unit="dega")
        if kind == "rotation":
            if angular_velocity is None:
                raise ToolError("A rotation requires 'angular_velocity'.")
            radians = _angular_velocity_rad_s(angular_velocity, unit)
            if angular_acceleration is not None:
                acceleration = _angular_acceleration_rad_s2(angular_acceleration, acceleration_unit)
            frame.AddRotationMotion(
                start_time=(start_time, "s"),
                stop_time=(stop_time, "s"),
                angular_velocity=(radians, "rad/s"),
                angular_acceleration=(
                    acceleration if acceleration is not None else (0.0, 0.0, 0.0),
                    "rad/s2",
                ),
            )
        elif kind == "vibration":
            if frequency_hz is None or amplitude is None or axis is None:
                raise ToolError(
                    "A vibration requires 'frequency_hz', 'amplitude', and 'direction'."
                )
            rate = 0.0 if amplitude_variation is None else float(amplitude_variation)
            frame.AddVibrationMotion(
                start_time=(start_time, "s"),
                stop_time=(stop_time, "s"),
                initial_frequency=(float(frequency_hz), "Hz"),
                initial_amplitude=(float(amplitude), "m"),
                direction=(tuple(axis), "m"),
                frequency_variation=(float(frequency_variation_hz_s), "Hz/s"),
                amplitude_variation=(rate, "m/s"),
            )
        elif kind == "pendulum":
            if frequency_hz is None or amplitude is None or axis is None:
                raise ToolError("A pendulum requires 'frequency_hz', 'amplitude', and 'direction'.")
            amplitude_rad = _angle_rad(amplitude, "dega", "amplitude")
            phase_rad = 0.0 if phase_deg is None else _angle_rad(phase_deg, "dega", "phase")
            amplitude_rate_rad_s = (
                0.0
                if amplitude_variation is None
                else _angle_rad(amplitude_variation, "dega", "amplitude_variation")
            )
            frame.AddPendulumMotion(
                start_time=(start_time, "s"),
                stop_time=(stop_time, "s"),
                initial_frequency=(float(frequency_hz), "Hz"),
                angular_initial_amplitude=(amplitude_rad, "rad"),
                direction=(tuple(axis), "m"),
                angular_initial_phase=(phase_rad, "rad"),
                frequency_variation=(float(frequency_variation_hz_s), "Hz/s"),
                angular_amplitude_variation=(amplitude_rate_rad_s, "rad/s"),
            )
        else:
            if linear_velocity is None:
                raise ToolError("A translation requires 'linear_velocity'.")
            if len(linear_velocity) != 3:
                raise ToolError("linear_velocity must be three numbers.")
            if linear_acceleration is not None:
                if len(linear_acceleration) != 3:
                    raise ToolError("linear_acceleration must be three numbers.")
                acceleration = tuple(float(value) for value in linear_acceleration)
            final: tuple[float, float, float] | None = None
            if final_velocity is not None:
                if len(final_velocity) != 3:
                    raise ToolError("final_velocity must be three numbers.")
                final = tuple(float(value) for value in final_velocity)
            chosen = _translation_definition(define_as, acceleration is not None, final is not None)
            frame.AddTranslationMotion(
                start_time=(start_time, "s"),
                stop_time=(stop_time, "s"),
                velocity=(tuple(float(value) for value in linear_velocity), "m/s"),
                acceleration=(
                    acceleration if acceleration is not None else (0.0, 0.0, 0.0),
                    "m/s2",
                ),
                define_as=chosen,
                final_velocity=(
                    final if final is not None else (0.0, 0.0, 0.0),
                    "m/s",
                ),
            )
        if period is not None:
            frame.SetPeriodicMotionPeriod(period, unit="s")
        if period_start is not None:
            frame.SetPeriodicMotionStartTime(period_start, unit="s")
        if period_stop is not None:
            frame.SetPeriodicMotionStopTime(period_stop, unit="s")
        if repeat is not None:
            frame.SetEnablePeriodicMotion(bool(repeat))
        if use_linear is not None:
            frame.SetEnableFbmLinearLimits(bool(use_linear))
        if min_linear is not None:
            frame.SetFbmMinLinearLimits(min_linear, unit="m")
        if max_linear is not None:
            frame.SetFbmMaxLinearLimits(max_linear, unit="m")
        if use_angular is not None:
            frame.SetEnableFbmAngularLimits(bool(use_angular))
        if min_angular is not None:
            frame.SetFbmMinAngularLimits(min_angular, unit="dega")
        if max_angular is not None:
            frame.SetFbmMaxAngularLimits(max_angular, unit="dega")
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
        angular_acceleration_rad_s2=(
            list(acceleration) if kind == "rotation" and acceleration is not None else None
        ),
        linear_acceleration_m_s2=(
            list(acceleration) if kind == "translation" and acceleration is not None else None
        ),
        define_as=chosen if kind == "translation" else None,
        origin_m=frame.GetRelativePosition("m") if placed is not None else None,
        orientation_deg=frame.GetOrientationFromAngles("dega") if angles is not None else None,
        period_s=frame.GetPeriodicMotionPeriod("s"),
        periodic=frame.GetEnablePeriodicMotion() if repeat is not None else None,
        linear_limits=frame.GetEnableFbmLinearLimits() if use_linear is not None else None,
        angular_limits=frame.GetEnableFbmAngularLimits() if use_angular is not None else None,
        min_linear_m=frame.GetFbmMinLinearLimits("m") if min_linear is not None else None,
        max_linear_m=frame.GetFbmMaxLinearLimits("m") if max_linear is not None else None,
        min_angular_deg=(frame.GetFbmMinAngularLimits("dega") if min_angular is not None else None),
        max_angular_deg=(frame.GetFbmMaxAngularLimits("dega") if max_angular is not None else None),
        period_start_s=frame.GetPeriodicMotionStartTime("s") if period_start is not None else None,
        period_stop_s=frame.GetPeriodicMotionStopTime("s") if period_stop is not None else None,
        frequency_hz=frequency_hz if kind in ("vibration", "pendulum") else None,
        amplitude_m=amplitude if kind == "vibration" else None,
        amplitude_rad=amplitude_rad,
        phase_rad=phase_rad,
        direction_m=axis if kind in ("vibration", "pendulum") else None,
        amplitude_variation_rad_s=amplitude_rate_rad_s,
    )


async def rocky_set_gravity(
    ctx: Context,
    gravity: list[float],
    unit: str = "m/s2",
    start_time: float | None = None,
    stop_time: float | None = None,
) -> dict[str, Any]:
    """Set the gravity vector on the study physics.

    Parameters
    ----------
    gravity : list[float]
        Gravity as ``[x, y, z]``.
    unit : str, default: "m/s2"
        Unit of the gravity components. Rocky's token is ``m/s2``.
    start_time : float | None, default: None
        Time in seconds when gravity starts. Rocky's token is ``s``.
    stop_time : float | None, default: None
        Time in seconds when gravity stops. Rocky's token is ``s``.

    """
    session = get_session(ctx)
    study = require_study(session)
    if len(gravity) != 3:
        raise ToolError("Gravity requires three components [x, y, z].")
    if unit != "m/s2":
        raise ToolError(f"Unknown gravity unit {unit!r}. Expected 'm/s2'.")
    try:
        physics = study.GetPhysics()
        physics.SetGravityXDirection(gravity[0], unit="m/s2")
        physics.SetGravityYDirection(gravity[1], unit="m/s2")
        physics.SetGravityZDirection(gravity[2], unit="m/s2")
        if start_time is not None:
            physics.SetGravityStartTime(start_time, unit="s")
        if stop_time is not None:
            physics.SetGravityStopTime(stop_time, unit="s")
        applied = [
            physics.GetGravityXDirection("m/s2"),
            physics.GetGravityYDirection("m/s2"),
            physics.GetGravityZDirection("m/s2"),
        ]
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Could not set gravity: {exc}") from exc
    return envelope(
        session,
        message="Gravity updated.",
        gravity_m_s2=applied,
        unit="m/s2",
        start_time_s=physics.GetGravityStartTime("s") if start_time is not None else None,
        stop_time_s=physics.GetGravityStopTime("s") if stop_time is not None else None,
    )


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
    thermal_correction: str | None = None,
    numerical_softening_factor: float | None = None,
    coarse_grain: bool | None = None,
    search_distance_multiplier: float | None = None,
    update_distance_multiplier: float | None = None,
    impact_energy: str | None = None,
    sph_thermal: str | None = None,
    use_radl: bool | None = None,
    exponent_limit: float | None = None,
    volume_fraction_limit: float | None = None,
    close_packing_volume_fraction: float | None = None,
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
    thermal_correction : str | None, default: None
        One of ``none``, ``morris_et_al_area``, or ``morris_et_al_area_time``.
    numerical_softening_factor : float | None, default: None
        Softens the contact stiffness used by the timestep.
    coarse_grain : bool | None, default: None
        Enable or disable coarse-grain modeling.
    search_distance_multiplier : float | None, default: None
        Multiplier on the neighbor-search distance.
    update_distance_multiplier : float | None, default: None
        Multiplier on the neighbor-list update distance.
    impact_energy : str | None, default: None
        One of ``default`` or ``custom``.
    sph_thermal : str | None, default: None
        SPH heat transfer. One of ``cleary`` or ``custom``.
    use_radl : bool | None, default: None
        Enable or disable the Radl et al. model.
    exponent_limit : float | None, default: None
        Exponent limit used by the contact model.
    volume_fraction_limit : float | None, default: None
        Volume-fraction limit.
    close_packing_volume_fraction : float | None, default: None
        Close-packing volume fraction.

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
    if thermal_correction is not None and thermal_correction not in THERMAL_CORRECTION_MODELS:
        raise ToolError(
            f"Unknown thermal correction {thermal_correction!r}. "
            f"Expected one of {THERMAL_CORRECTION_MODELS}."
        )
    if impact_energy is not None and impact_energy not in IMPACT_ENERGY_MODELS:
        raise ToolError(
            f"Unknown impact energy model {impact_energy!r}. "
            f"Expected one of {IMPACT_ENERGY_MODELS}."
        )
    if sph_thermal is not None and sph_thermal not in SPH_THERMAL_MODELS:
        raise ToolError(
            f"Unknown SPH thermal model {sph_thermal!r}. Expected one of {SPH_THERMAL_MODELS}."
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
        if thermal_correction is not None:
            physics.SetThermalCorrectionModel(thermal_correction)
        if numerical_softening_factor is not None:
            physics.SetNumericalSofteningFactor(numerical_softening_factor)
        if coarse_grain is not None:
            physics.SetEnableCoarseGrainModeling(coarse_grain)
        if search_distance_multiplier is not None:
            physics.SetSearchDistanceMultiplier(search_distance_multiplier)
        if update_distance_multiplier is not None:
            physics.SetUpdateDistanceMultiplier(update_distance_multiplier)
        if impact_energy is not None:
            physics.SetImpactEnergyModel(impact_energy)
        if sph_thermal is not None:
            physics.SetSphThermalTransferModel(sph_thermal)
        if use_radl is not None:
            physics.SetUseRadlEtAl(use_radl)
        if exponent_limit is not None:
            physics.SetExponentLimit(exponent_limit)
        if volume_fraction_limit is not None:
            physics.SetVolumeFractionLimit(volume_fraction_limit)
        if close_packing_volume_fraction is not None:
            physics.SetClosePackingVolumeFraction(close_packing_volume_fraction)
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
        thermal_correction=physics.GetThermalCorrectionModel(),
        numerical_softening_factor=physics.GetNumericalSofteningFactor(),
        coarse_grain=bool(physics.GetEnableCoarseGrainModeling()),
        search_distance_multiplier=physics.GetSearchDistanceMultiplier(),
        update_distance_multiplier=physics.GetUpdateDistanceMultiplier(),
        impact_energy=physics.GetImpactEnergyModel(),
        sph_thermal=physics.GetSphThermalTransferModel(),
        use_radl=bool(physics.GetUseRadlEtAl()),
        exponent_limit=physics.GetExponentLimit(),
        volume_fraction_limit=physics.GetVolumeFractionLimit(),
        close_packing_volume_fraction=physics.GetClosePackingVolumeFraction(),
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


_THERMAL_BOUNDARIES = ("adiabatic", "prescribed_temperature", "cfd_coupled_temperature")


async def rocky_set_temperature(
    ctx: Context,
    object_id: str,
    temperature: float,
    thermal_boundary: str | None = None,
) -> dict[str, Any]:
    """Set a temperature in kelvin on a particle inlet, wall, or conveyor.

    Parameters
    ----------
    object_id : str
        Identifier of a particle inlet, wall, or conveyor.
    temperature : float
        Temperature in kelvin. Rocky's token is ``K``.
    thermal_boundary : str | None, default: None
        For a wall: ``adiabatic``, ``prescribed_temperature``, or
        ``cfd_coupled_temperature``. A wall temperature selects
        ``prescribed_temperature``.

    """
    session = get_session(ctx)
    require_study(session)
    entry = session.registry.get(object_id)
    if thermal_boundary is not None and thermal_boundary not in _THERMAL_BOUNDARIES:
        raise ToolError(
            f"Unknown thermal boundary {thermal_boundary!r}. Expected one of {_THERMAL_BOUNDARIES}."
        )
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
            if entry.category == "wall":
                boundary = thermal_boundary or "prescribed_temperature"
                setter = getattr(entry.obj, "SetThermalBoundaryConditionType", None)
                if setter is not None:
                    setter(boundary)
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
        thermal_boundary=getattr(entry.obj, "GetThermalBoundaryConditionType", lambda: None)(),
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
    mcp.tool()(rocky_add_inlet_particle)
    mcp.tool()(rocky_remove_inlet_particle)
    mcp.tool()(rocky_create_volumetric_inlet)
    mcp.tool()(rocky_create_outlet)
    mcp.tool()(rocky_set_domain)
    mcp.tool()(rocky_set_simulation_settings)
    mcp.tool()(rocky_set_output_property)
    mcp.tool()(rocky_set_sph_boundary)
    mcp.tool()(rocky_set_wall_wear)
    mcp.tool()(rocky_set_wall_mass)
    mcp.tool()(rocky_set_gravity)
    mcp.tool()(rocky_set_contact_models)
    mcp.tool()(rocky_set_motion)
    mcp.tool()(rocky_set_temperature)
