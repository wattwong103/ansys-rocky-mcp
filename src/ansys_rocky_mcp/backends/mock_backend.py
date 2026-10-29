"""In-memory mock of the Ansys Rocky application.

This module implements the subset of the Rocky API used by the MCP server.
It is a *simulator*, not a DEM solver: it models the bookkeeping of a Rocky
study (project, study, particles, surfaces, inlets, outlets, selections,
domain and solver settings) and produces plausible, physically-consistent
time series for mass flow, particle count and velocity.

Every response produced through this backend is tagged with
``backend="mock"`` and ``simulated=true`` so that simulated numbers can never
be mistaken for a real Rocky run.
"""

from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any

from ansys_rocky_mcp.backends.base import BackendStatus, RockyBackend

__all__ = ["MockBackend", "MockRockyApplication"]

#: Bulk density of a typical granular bed (kg/m3).
BULK_DENSITY = 1600.0

#: Density of the solid particles (kg/m3).
PARTICLE_DENSITY = 2500.0

#: Default sieve size of a particle (m).
DEFAULT_SIEVE_SIZE = 0.05

#: Default inlet radius (m).
DEFAULT_INLET_RADIUS = 0.3

#: Default domain extent along X (m).
DEFAULT_DOMAIN_LENGTH = 10.0

#: Default simulation duration (s).
DEFAULT_DURATION = 2.0

_LENGTH_TO_M = {"m": 1.0, "cm": 0.01, "mm": 0.001, "km": 1000.0}
_MASS_FLOW_TO_KG_S = {"kg/s": 1.0, "t/h": 1.0 / 3.6, "kg/h": 1.0 / 3600.0}
_VELOCITY_TO_M_S = {"m/s": 1.0, "km/h": 1.0 / 3.6, "mm/s": 0.001}


def _length_to_m(value: float, unit: str | None) -> float:
    """Convert a length to meters."""
    if unit is None:
        return float(value)
    try:
        return float(value) * _LENGTH_TO_M[unit]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported length unit {unit!r}. Expected one of {sorted(_LENGTH_TO_M)}."
        ) from exc


def _mass_flow_to_kg_s(value: float, unit: str | None) -> float:
    """Convert a mass flow rate to kilograms per second."""
    if unit is None:
        return float(value) / 3.6  # Rocky's default mass-flow unit is t/h.
    try:
        return float(value) * _MASS_FLOW_TO_KG_S[unit]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported mass-flow unit {unit!r}. Expected one of {sorted(_MASS_FLOW_TO_KG_S)}."
        ) from exc


def _velocity_to_m_s(value: float, unit: str | None) -> float:
    """Convert a velocity to meters per second."""
    if unit is None:
        return float(value)
    try:
        return float(value) * _VELOCITY_TO_M_S[unit]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported velocity unit {unit!r}. Expected one of {sorted(_VELOCITY_TO_M_S)}."
        ) from exc


def _as_sequence(value: Any, count: int = 3) -> tuple[float, ...]:
    """Normalize a scalar-sequence argument into a tuple of ``count`` floats."""
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = [value]
    if len(items) == 1 and count > 1:
        items = items * count
    if len(items) != count:
        raise ValueError(f"Expected {count} values, got {len(items)}.")
    return tuple(float(item) for item in items)


class _VectorMixin:
    """Shared center handling for surfaces and selection processes."""

    def _init_center(self) -> None:
        self._center: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def SetCenter(self, *args: Any, unit: str | None = None, **kwargs: Any) -> None:
        """Set the center, accepting either a sequence or three scalars."""
        if args and kwargs:
            raise TypeError("Pass the center either positionally or as 'values'.")
        values = kwargs.pop("values", None)
        if values is None:
            values = args[0] if len(args) == 1 else args
        x, y, z = _as_sequence(values)
        factor = _LENGTH_TO_M.get(unit or "m", 1.0)
        self._center = (x * factor, y * factor, z * factor)

    def GetCenter(self, unit: str | None = None) -> list[float]:
        """Return the center in the requested unit."""
        factor = 1.0 / _LENGTH_TO_M.get(unit or "m", 1.0)
        return [component * factor for component in self._center]


class MockGridFunction:
    """Stand-in for a Rocky grid function such as ``Velocity : Translational : Absolute``."""

    def __init__(self, study: MockStudy, name: str) -> None:
        """Initialize the grid function bound to a study."""
        self._study = study
        self._name = name

    def GetName(self) -> str:
        """Return the grid function name."""
        return self._name

    def _value(self, time_step: Any, kind: str) -> float:
        time = self._study._resolve_time(time_step)
        velocity = self._study._derived()["velocity"]
        period = max(self._study._derived()["transit"] / 3.0, 0.5)
        ramp = min(1.0, time / max(0.1 * self._study._derived()["transit"], 1e-9))
        oscillation = 1.0 + 0.12 * math.sin(2.0 * math.pi * time / period)
        magnitude = velocity * oscillation * ramp
        if kind == "max":
            return magnitude
        return magnitude * 0.05

    def GetMax(self, unit: str | None = None, time_step: Any = "current") -> float:
        """Return the maximum value of the grid function at a time step."""
        return _velocity_to_m_s(self._value(time_step, "max"), unit)

    def GetMin(self, unit: str | None = None, time_step: Any = "current") -> float:
        """Return the minimum value of the grid function at a time step."""
        return _velocity_to_m_s(self._value(time_step, "min"), unit)

    def GetAverage(self, unit: str | None = None, time_step: Any = "current") -> float:
        """Return the average value of the grid function at a time step."""
        value = (self._value(time_step, "max") + self._value(time_step, "min")) / 2.0
        return _velocity_to_m_s(value, unit)


class MockParticlesProcess:
    """Stand-in for the ``Particles`` output object returned by ``study.GetParticles()``."""

    GRID_FUNCTION_NAMES = (
        "Velocity : Translational : Absolute",
        "Velocity : Translational : X",
        "Velocity : Translational : Y",
        "Velocity : Translational : Z",
        "Solid Fraction",
        "Total Mass",
    )

    def __init__(self, study: MockStudy) -> None:
        """Initialize the process bound to a study."""
        self._study = study

    def GetNumpyCurve(self, curve_name: str, unit: str | None = None) -> tuple[Any, Any]:
        """Return ``(times, values)`` for a named curve."""
        times, values, base_unit, default_unit = self._study._curve_data(curve_name)
        if base_unit is not None:
            table = _MASS_FLOW_TO_KG_S if base_unit == "kg/s" else _VELOCITY_TO_M_S
            target = unit or default_unit
            if target not in table:
                raise ValueError(
                    f"Unsupported unit {target!r} for the curve {curve_name!r}. "
                    f"Expected one of {sorted(table)}."
                )
            factor = 1.0 / table[target]
            values = [value * factor for value in values]
        return times, values

    def GetCurve(self, curve_name: str, **kwargs: Any) -> tuple[Any, Any]:
        """Return ``(times, values)`` for a named curve."""
        return self.GetNumpyCurve(curve_name)

    def GetGridFunction(
        self,
        grid_function_name: str,
        simulation_name: str | None = None,
        translated: bool = False,
    ) -> MockGridFunction:
        """Return a grid function object."""
        return MockGridFunction(self._study, grid_function_name)

    def GetGridFunctionNames(
        self, translated: bool = False, context: str | None = None
    ) -> list[str]:
        """Return the available grid function names."""
        return list(self.GRID_FUNCTION_NAMES)

    def GetNumberOfParticles(self, time_step: Any = "current") -> int:
        """Return the number of particles at a time step."""
        time = self._study._resolve_time(time_step)
        return int(self._study._particle_count(time))


class MockParticle:
    """Stand-in for a Rocky particle definition."""

    def __init__(self, name: str = "Particle") -> None:
        """Initialize the particle."""
        self._name = name
        self._shape = "sphere"
        self._size_type = "sieve"
        self._minimum_size = DEFAULT_SIEVE_SIZE

    def GetName(self) -> str:
        """Return the particle name."""
        return self._name

    def GetClassName(self) -> str:
        """Return the Rocky class name of the particle."""
        return "RAParticle"

    def SetName(self, value: str) -> None:
        """Set the particle name."""
        self._name = value

    def GetShape(self) -> str:
        """Return the particle shape."""
        return self._shape

    def SetShape(self, value: str) -> None:
        """Set the particle shape."""
        self._shape = value

    def GetSizeType(self) -> str:
        """Return the particle size type."""
        return self._size_type

    def SetSizeType(self, value: str) -> None:
        """Set the particle size type."""
        self._size_type = value

    def GetMinimumSize(self, unit: str | None = None) -> float:
        """Return the sieve size in the requested unit."""
        factor = 1.0 / _LENGTH_TO_M.get(unit or "m", 1.0)
        return self._minimum_size * factor

    def SetMinimumSize(self, value: float, unit: str | None = None) -> None:
        """Set the sieve size."""
        self._minimum_size = _length_to_m(value, unit)


class MockParticleInletProperties:
    """Stand-in for a particle inlet property set."""

    def __init__(self, particle: MockParticle) -> None:
        """Initialize the properties for a particle."""
        self._particle = particle
        self._mass_flow_tph = 0.0

    def GetParticle(self) -> MockParticle:
        """Return the particle bound to these properties."""
        return self._particle

    def GetMassFlowRate(self, unit: str | None = None) -> float:
        """Return the mass flow rate in the requested unit."""
        kg_s = self._mass_flow_tph / 3.6
        if unit in (None, "t/h"):
            return kg_s * 3.6
        try:
            return kg_s / _MASS_FLOW_TO_KG_S[unit]
        except KeyError as exc:
            raise ValueError(f"Unsupported mass-flow unit {unit!r}.") from exc

    def SetMassFlowRate(self, value: float, unit: str | None = None) -> None:
        """Set the mass flow rate."""
        self._mass_flow_tph = _mass_flow_to_kg_s(value, unit) * 3.6


class MockParticleInlet:
    """Stand-in for a Rocky particle inlet."""

    def __init__(self, entry_point: Any, particle: MockParticle) -> None:
        """Initialize the inlet."""
        self._entry_point = entry_point
        self._particle = particle
        self._name = "Particle Inlet"
        self._properties = MockParticleInletProperties(particle)
        self._injection_duration = 0.0

    def GetName(self) -> str:
        """Return the inlet name."""
        return self._name

    def GetClassName(self) -> str:
        """Return the Rocky class name of the inlet."""
        return "RAParticleInlet"

    def SetName(self, value: str) -> None:
        """Set the inlet name."""
        self._name = value

    def GetEntryPoint(self) -> Any:
        """Return the entry point (surface) of the inlet."""
        return self._entry_point

    def GetParticle(self) -> MockParticle:
        """Return the particle fed by the inlet."""
        return self._particle

    def GetInputPropertiesList(self) -> list[MockParticleInletProperties]:
        """Return the inlet property list."""
        return [self._properties]

    def GetInjectionDuration(self, unit: str | None = None) -> float:
        """Return the injection duration."""
        return self._injection_duration

    def SetInjectionDuration(self, value: float, unit: str | None = None) -> None:
        """Set the injection duration."""
        self._injection_duration = _length_to_m(value, unit)


class MockOutlet:
    """Stand-in for a Rocky outlet."""

    def __init__(self, exit_point: Any) -> None:
        """Initialize the outlet."""
        self._exit_point = exit_point
        self._name = "Outlet"

    def GetName(self) -> str:
        """Return the outlet name."""
        return self._name

    def GetClassName(self) -> str:
        """Return the Rocky class name of the outlet."""
        return "RAOutlet"

    def SetName(self, value: str) -> None:
        """Set the outlet name."""
        self._name = value

    def GetExitPoint(self) -> Any:
        """Return the exit point (surface) of the outlet."""
        return self._exit_point


class MockSurface(_VectorMixin):
    """Stand-in for a circular or rectangular Rocky surface."""

    def __init__(self, kind: str = "circular") -> None:
        """Initialize the surface."""
        self._init_center()
        self._kind = kind
        self._name = f"{kind.title()} Surface"
        self._max_radius = DEFAULT_INLET_RADIUS
        self._min_radius = 0.0
        self._length = 1.0
        self._width = 1.0

    def GetName(self) -> str:
        """Return the surface name."""
        return self._name

    def SetName(self, value: str) -> None:
        """Set the surface name."""
        self._name = value

    def GetKind(self) -> str:
        """Return the surface kind (``circular`` or ``rectangular``)."""
        return self._kind

    def GetClassName(self) -> str:
        """Return the Rocky class name of the surface."""
        return "RACircularSurface" if self._kind == "circular" else "RARectangularSurface"

    def GetMaxRadius(self, unit: str | None = None) -> float:
        """Return the maximum radius."""
        return self._max_radius / _LENGTH_TO_M.get(unit or "m", 1.0)

    def SetMaxRadius(self, value: float, unit: str | None = None) -> None:
        """Set the maximum radius."""
        self._max_radius = _length_to_m(value, unit)

    def GetMinRadius(self, unit: str | None = None) -> float:
        """Return the minimum radius."""
        return self._min_radius / _LENGTH_TO_M.get(unit or "m", 1.0)

    def SetMinRadius(self, value: float, unit: str | None = None) -> None:
        """Set the minimum radius."""
        self._min_radius = _length_to_m(value, unit)

    def GetLength(self, unit: str | None = None) -> float:
        """Return the length of a rectangular surface."""
        return self._length / _LENGTH_TO_M.get(unit or "m", 1.0)

    def SetLength(self, value: float, unit: str | None = None) -> None:
        """Set the length of a rectangular surface."""
        self._length = _length_to_m(value, unit)

    def GetWidth(self, unit: str | None = None) -> float:
        """Return the width of a rectangular surface."""
        return self._width / _LENGTH_TO_M.get(unit or "m", 1.0)

    def SetWidth(self, value: float, unit: str | None = None) -> None:
        """Set the width of a rectangular surface."""
        self._width = _length_to_m(value, unit)


class MockWall:
    """Stand-in for an imported Rocky wall."""

    def __init__(self, filename: str, name: str) -> None:
        """Initialize the wall."""
        self._filename = filename
        self._name = name

    def GetName(self) -> str:
        """Return the wall name."""
        return self._name

    def GetClassName(self) -> str:
        """Return the Rocky class name of the wall."""
        return "RAWall"

    def GetFilename(self) -> str:
        """Return the source file of the wall."""
        return self._filename


class MockDomainSettings:
    """Stand-in for Rocky domain settings."""

    def __init__(self) -> None:
        """Initialize the domain."""
        self._use_boundary_limits = True
        self._min = (-DEFAULT_DOMAIN_LENGTH / 2.0, -1.0, -1.0)
        self._max = (DEFAULT_DOMAIN_LENGTH / 2.0, 1.0, 1.0)
        self._domain_type = "rectangular"

    def GetUseBoundaryLimits(self) -> bool:
        """Return whether boundary limits are used."""
        return self._use_boundary_limits

    def EnableUseBoundaryLimits(self) -> None:
        """Enable boundary limits."""
        self._use_boundary_limits = True

    def DisableUseBoundaryLimits(self) -> None:
        """Disable boundary limits."""
        self._use_boundary_limits = False

    def GetCoordinateLimitsMinValues(self, unit: str | None = None) -> list[float]:
        """Return the minimum domain coordinates."""
        factor = 1.0 / _LENGTH_TO_M.get(unit or "m", 1.0)
        return [value * factor for value in self._min]

    def SetCoordinateLimitsMinValues(
        self, values: Any, unit: str | None = None, **kwargs: Any
    ) -> None:
        """Set the minimum domain coordinates."""
        factor = _LENGTH_TO_M.get(unit or "m", 1.0)
        self._min = tuple(value * factor for value in _as_sequence(values))

    def GetCoordinateLimitsMaxValues(self, unit: str | None = None) -> list[float]:
        """Return the maximum domain coordinates."""
        factor = 1.0 / _LENGTH_TO_M.get(unit or "m", 1.0)
        return [value * factor for value in self._max]

    def SetCoordinateLimitsMaxValues(
        self, values: Any, unit: str | None = None, **kwargs: Any
    ) -> None:
        """Set the maximum domain coordinates."""
        factor = _LENGTH_TO_M.get(unit or "m", 1.0)
        self._max = tuple(value * factor for value in _as_sequence(values))

    def GetCoordinateLimitsMinMax(self) -> tuple[list[float], list[float]]:
        """Return the ``(min, max)`` domain coordinates."""
        return list(self._min), list(self._max)

    def GetDomainType(self) -> str:
        """Return the domain type."""
        return self._domain_type

    def SetDomainType(self, value: str) -> None:
        """Set the domain type."""
        self._domain_type = value


class MockSimulatorRun:
    """Stand-in for the Rocky solver / simulator run object."""

    def __init__(self) -> None:
        """Initialize the solver."""
        self._duration = DEFAULT_DURATION
        self._time_interval = DEFAULT_DURATION / 20.0
        self._processors = 1
        self._execution_mode = "parallel"
        self._gpu = 0

    def GetSimulationDuration(self, unit: str | None = None) -> float:
        """Return the simulation duration."""
        return self._duration

    def SetSimulationDuration(self, value: float, unit: str | None = None) -> None:
        """Set the simulation duration."""
        self._duration = float(value)

    def GetTimeInterval(self, unit: str | None = None) -> float:
        """Return the output time interval."""
        return self._time_interval

    def SetTimeInterval(self, value: float, unit: str | None = None) -> None:
        """Set the output time interval."""
        self._time_interval = float(value)

    def SetTimeConfiguration(self, duration: float, time_interval: float) -> None:
        """Set the duration and output interval together."""
        self._duration = float(duration)
        self._time_interval = float(time_interval)

    def GetNumberOfProcessors(self) -> int:
        """Return the number of processors."""
        return self._processors

    def SetNumberOfProcessors(self, value: int | str) -> None:
        """Set the number of processors."""
        self._processors = int(value)

    def GetExecutionMode(self) -> str:
        """Return the execution mode."""
        return self._execution_mode

    def SetExecutionMode(self, value: str) -> None:
        """Set the execution mode."""
        self._execution_mode = value

    def GetValidExecutionModeValues(self) -> list[str]:
        """Return the supported execution modes."""
        return ["parallel", "serial"]

    def GetTargetGpu(self) -> int:
        """Return the target GPU index."""
        return self._gpu

    def SetTargetGpu(self, value: int) -> None:
        """Set the target GPU index."""
        self._gpu = int(value)

    def GetFixedTimestep(self, unit: str | None = None) -> float:
        """Return the fixed timestep."""
        return 1e-4

    def SetFixedTimestep(self, value: float, unit: str | None = None) -> None:
        """Set the fixed timestep (recorded but not used by the mock)."""


class MockMotionFrame:
    """Stand-in for a Rocky motion frame."""

    def __init__(self) -> None:
        """Initialize the frame."""
        self._motions: list[dict[str, Any]] = []
        self._applied_to: list[str] = []

    def AddRotationMotion(
        self,
        start_time: Any = (0.0, "s"),
        stop_time: Any = (1000.0, "s"),
        angular_velocity: Any = ((0.0, 0.0, 0.0), "rad/s"),
        angular_acceleration: Any = ((0.0, 0.0, 0.0), "rad/s2"),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a rotation motion to the frame."""
        motion = {
            "kind": "rotation",
            "start_time": list(start_time) if isinstance(start_time, tuple) else start_time,
            "stop_time": list(stop_time) if isinstance(stop_time, tuple) else stop_time,
            "angular_velocity": list(angular_velocity),
            "angular_acceleration": list(angular_acceleration),
        }
        self._motions.append(motion)
        return motion

    def AddTranslationMotion(
        self,
        start_time: Any = (0.0, "s"),
        stop_time: Any = (1000.0, "s"),
        velocity: Any = ((0.0, 0.0, 0.0), "m/s"),
        acceleration: Any = ((0.0, 0.0, 0.0), "m/s2"),
        define_as: str = "fixed_velocity",
        final_velocity: Any = ((0.0, 0.0, 0.0), "m/s"),
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add a translation motion to the frame."""
        motion = {
            "kind": "translation",
            "start_time": list(start_time) if isinstance(start_time, tuple) else start_time,
            "stop_time": list(stop_time) if isinstance(stop_time, tuple) else stop_time,
            "velocity": list(velocity),
            "acceleration": list(acceleration),
            "define_as": define_as,
            "final_velocity": list(final_velocity),
        }
        self._motions.append(motion)
        return motion

    def ApplyTo(self, obj: Any) -> None:
        """Apply the frame to an object."""
        name = getattr(obj, "GetName", lambda: str(obj))()
        self._applied_to.append(name)

    def GetMotions(self) -> list[dict[str, Any]]:
        """Return the motions defined in the frame."""
        return list(self._motions)

    def GetAppliedTo(self) -> list[str]:
        """Return the names of the objects the frame was applied to."""
        return list(self._applied_to)


class MockMotionFrameSource:
    """Stand-in for the Rocky motion frame source."""

    def __init__(self) -> None:
        """Initialize the source."""
        self._frames: list[MockMotionFrame] = []

    def NewFrame(self) -> MockMotionFrame:
        """Create a new motion frame."""
        frame = MockMotionFrame()
        self._frames.append(frame)
        return frame


class MockSelectionProcess(_VectorMixin):
    """Base class for selection processes such as cylinders and cubes."""

    def __init__(self, name: str) -> None:
        """Initialize the selection."""
        self._init_center()
        self._name = name
        self._size = (1.0, 1.0, 1.0)
        self._final_angle = 360.0
        self._rotation = (0.0, 0.0, 0.0)

    def GetName(self) -> str:
        """Return the process name."""
        return self._name

    def GetClassName(self) -> str:
        """Return the Rocky class name of the selection."""
        return "RACylinderProcess" if isinstance(self, MockCylinderProcess) else "RACubeProcess"

    def SetName(self, value: str) -> None:
        """Set the process name."""
        self._name = value

    def GetSize(self, unit: str | None = None) -> list[float]:
        """Return the size of the selection."""
        factor = 1.0 / _LENGTH_TO_M.get(unit or "m", 1.0)
        return [value * factor for value in self._size]

    def SetSize(self, x: float, y: float, z: float, unit: str | None = None) -> None:
        """Set the size of the selection."""
        factor = _LENGTH_TO_M.get(unit or "m", 1.0)
        self._size = (x * factor, y * factor, z * factor)

    def SetFinalAngle(self, value: float, unit: str | None = None) -> None:
        """Set the final angle of the selection."""
        self._final_angle = float(value)

    def GetFinalAngle(self, unit: str | None = None) -> float:
        """Return the final angle of the selection."""
        return self._final_angle

    def SetRotation(self, x: float, y: float, z: float, unit: str | None = None) -> None:
        """Set the rotation of the selection."""
        self._rotation = (x, y, z)

    def GetRotation(self, unit: str | None = None) -> list[float]:
        """Return the rotation of the selection."""
        return list(self._rotation)

    def GetNumpyCurve(self, curve_name: str, unit: str | None = None) -> tuple[Any, Any]:
        """Return the curve restricted to the selection."""
        study = self._study
        times, values, base_unit, default_unit = study._curve_data(curve_name)
        if base_unit is not None:
            table = _MASS_FLOW_TO_KG_S if base_unit == "kg/s" else _VELOCITY_TO_M_S
            target = unit or default_unit
            if target not in table:
                raise ValueError(
                    f"Unsupported unit {target!r} for the curve {curve_name!r}. "
                    f"Expected one of {sorted(table)}."
                )
            factor = 1.0 / table[target]
            values = [value * factor for value in values]
        fraction = self._fraction()
        return times, [value * fraction for value in values]

    def GetCurve(self, curve_name: str, **kwargs: Any) -> tuple[Any, Any]:
        """Return the curve restricted to the selection."""
        return self.GetNumpyCurve(curve_name)

    def GetNumberOfParticles(self, time_step: Any = "current") -> int:
        """Return the number of particles inside the selection."""
        study = self._study
        time = study._resolve_time(time_step)
        return int(study._particle_count(time) * self._fraction())

    def RemoveProcess(self) -> None:
        """Remove the process from the project."""
        if self._study is not None:
            self._study._remove_selection(self)

    def _fraction(self) -> float:
        """Return the fraction of the domain covered by the selection."""
        study = self._study
        domain = study._domain
        min_corner, max_corner = domain.GetCoordinateLimitsMinMax()
        domain_volume = 1.0
        for low, high in zip(min_corner, max_corner, strict=False):
            domain_volume *= max(high - low, 1e-9)
        selection_volume = 1.0
        for extent in self._size:
            selection_volume *= max(extent, 1e-9)
        return min(selection_volume / domain_volume, 1.0)


class MockUserProcessCollection:
    """Stand-in for the Rocky user process collection."""

    def __init__(self, study: MockStudy) -> None:
        """Initialize the collection."""
        self._study = study

    def GetProcessNames(self) -> list[str]:
        """Return the names of all user processes."""
        return [process.GetName() for process in self._study._selections]

    def GetCylinderProcessNames(self) -> list[str]:
        """Return the names of the cylinder processes."""
        return [
            process.GetName()
            for process in self._study._selections
            if isinstance(process, MockCylinderProcess)
        ]

    def CreateCylinderProcess(
        self,
        parent: Any = None,
        name: str | None = None,
        center: Any = None,
        size: Any = None,
        rotation: Any = None,
        internal_factor: float | None = None,
        initial_angle: float | None = None,
        final_angle: float | None = None,
        **kwargs: Any,
    ) -> MockCylinderProcess:
        """Create a cylinder selection process."""
        count = len(self._study._selections) + 1
        process = MockCylinderProcess(name or f"Cylinder Selection {count}")
        process._study = self._study
        if center is not None:
            process.SetCenter(center)
        if size is not None:
            sx, sy, sz = _as_sequence(size)
            process.SetSize(sx, sy, sz)
        if rotation is not None:
            rx, ry, rz = _as_sequence(rotation)
            process.SetRotation(rx, ry, rz)
        if final_angle is not None:
            process.SetFinalAngle(final_angle)
        self._study._selections.append(process)
        return process

    def CreateCubeProcess(
        self,
        parent: Any = None,
        name: str | None = None,
        center: Any = None,
        size: Any = None,
        **kwargs: Any,
    ) -> MockCubeProcess:
        """Create a cube selection process."""
        process = MockCubeProcess(name or f"Cube Selection {len(self._study._selections) + 1}")
        process._study = self._study
        if center is not None:
            process.SetCenter(center)
        if size is not None:
            sx, sy, sz = _as_sequence(size)
            process.SetSize(sx, sy, sz)
        self._study._selections.append(process)
        return process

    def RemoveProcess(self, process: Any) -> None:
        """Remove a user process."""
        name = process if isinstance(process, str) else process.GetName()
        for candidate in list(self._study._selections):
            if candidate.GetName() == name:
                self._study._remove_selection(candidate)


class MockCylinderProcess(MockSelectionProcess):
    """Stand-in for a Rocky cylinder selection process."""

    def __init__(self, name: str) -> None:
        """Initialize the cylinder selection."""
        super().__init__(name)
        self._study: MockStudy | None = None


class MockCubeProcess(MockSelectionProcess):
    """Stand-in for a Rocky cube selection process."""

    def __init__(self, name: str) -> None:
        """Initialize the cube selection."""
        super().__init__(name)
        self._study: MockStudy | None = None


class MockStudy:
    """Stand-in for a Rocky study, the container of a DEM simulation."""

    def __init__(self, name: str = "Study 1") -> None:
        """Initialize an empty study."""
        self._name = name
        self._particles: list[MockParticle] = []
        self._surfaces: list[MockSurface] = []
        self._inlets: list[MockParticleInlet] = []
        self._outlets: list[MockOutlet] = []
        self._walls: list[MockWall] = []
        self._selections: list[MockSelectionProcess] = []
        self._domain = MockDomainSettings()
        self._solver = MockSimulatorRun()
        self._motion_source = MockMotionFrameSource()
        self._processes = MockUserProcessCollection(self)
        self._particles_process = MockParticlesProcess(self)

        self._simulating = False
        self._elapsed = 0.0
        self._progress = 0.0
        self._has_results = False
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._speed = 8.0

    # ------------------------------------------------------------------
    # Entity creation
    # ------------------------------------------------------------------
    def GetName(self) -> str:
        """Return the study name."""
        return self._name

    def GetClassName(self) -> str:
        """Return the Rocky class name of the study."""
        return "RAStudy"

    def CreateParticle(self) -> MockParticle:
        """Create a particle definition."""
        particle = MockParticle(f"Particle {len(self._particles) + 1}")
        self._particles.append(particle)
        return particle

    def CreateCircularSurface(self) -> MockSurface:
        """Create a circular surface."""
        surface = MockSurface("circular")
        self._surfaces.append(surface)
        return surface

    def CreateRectangularSurface(self) -> MockSurface:
        """Create a rectangular surface."""
        surface = MockSurface("rectangular")
        self._surfaces.append(surface)
        return surface

    def CreateParticleInlet(
        self, entry_point: Any = None, particle: Any = None, **kwargs: Any
    ) -> MockParticleInlet:
        """Create a particle inlet on a surface."""
        if entry_point is None:
            entry_point = kwargs.get("entry_point")
        if particle is None:
            particle = kwargs.get("particle")
        if entry_point is None or particle is None:
            raise ValueError("CreateParticleInlet requires an entry_point and a particle.")
        if not isinstance(entry_point, MockSurface):
            raise TypeError("The entry point of a particle inlet must be a surface.")
        if not isinstance(particle, MockParticle):
            raise TypeError("The particle of a particle inlet must be a particle definition.")
        inlet = MockParticleInlet(entry_point, particle)
        self._inlets.append(inlet)
        return inlet

    def CreateOutlet(self, exit_point: Any = None, **kwargs: Any) -> MockOutlet:
        """Create an outlet on a surface."""
        if exit_point is None:
            exit_point = kwargs.get("exit_point")
        if exit_point is None:
            raise ValueError("CreateOutlet requires an exit point.")
        if not isinstance(exit_point, MockSurface):
            raise TypeError("The exit point of an outlet must be a surface.")
        outlet = MockOutlet(exit_point)
        self._outlets.append(outlet)
        return outlet

    def ImportWall(
        self,
        custom_filename: str,
        import_scale: float = 1.0,
        convert_yz: bool = False,
        custom_name_prefix: str | None = None,
        **kwargs: Any,
    ) -> list[MockWall]:
        """Import a wall geometry."""
        path = Path(custom_filename)
        prefix = custom_name_prefix or path.stem
        wall = MockWall(str(path), f"{prefix}")
        self._walls.append(wall)
        return [wall]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def GetDomainSettings(self) -> MockDomainSettings:
        """Return the domain settings."""
        return self._domain

    def GetSolver(self) -> MockSimulatorRun:
        """Return the solver."""
        return self._solver

    def GetSimulatorRun(self) -> MockSimulatorRun:
        """Return the simulator run."""
        return self._solver

    def GetMotionFrameSource(self) -> MockMotionFrameSource:
        """Return the motion frame source."""
        return self._motion_source

    def GetParticles(self) -> MockParticlesProcess:
        """Return the particles output object."""
        return self._particles_process

    def GetUserProcessCollection(self) -> MockUserProcessCollection:
        """Return the user process collection."""
        return self._processes

    def GetExportToolkit(self) -> Any:
        """Return an export toolkit stub."""
        raise NotImplementedError("The mock backend does not implement result export.")

    def GetElementNames(self, type_name: str | None = None) -> list[str]:
        """Return the names of the study elements."""
        names: list[str] = []
        if type_name in (None, "Particle"):
            names.extend(particle.GetName() for particle in self._particles)
        if type_name in (None, "Surface"):
            names.extend(surface.GetName() for surface in self._surfaces)
        if type_name in (None, "Inlet"):
            names.extend(inlet.GetName() for inlet in self._inlets)
        if type_name in (None, "Outlet"):
            names.extend(outlet.GetName() for outlet in self._outlets)
        if type_name in (None, "Wall"):
            names.extend(wall.GetName() for wall in self._walls)
        return names

    def GetElement(
        self,
        element_name: str | None = None,
        type_name: str | None = None,
        raise_if_no_found: bool = True,
    ) -> Any:
        """Return a study element by name."""
        candidates: list[Any] = []
        if type_name in (None, "Particle"):
            candidates.extend(self._particles)
        if type_name in (None, "Surface"):
            candidates.extend(self._surfaces)
        if type_name in (None, "Inlet"):
            candidates.extend(self._inlets)
        if type_name in (None, "Outlet"):
            candidates.extend(self._outlets)
        if type_name in (None, "Wall"):
            candidates.extend(self._walls)
        for candidate in candidates:
            if element_name is None or candidate.GetName() == element_name:
                return candidate
        if raise_if_no_found:
            raise KeyError(f"No study element named {element_name!r}.")
        return None

    # ------------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------------
    def SetMockSpeed(self, speed: float) -> None:
        """Set the simulated seconds per wall-clock second."""
        self._speed = max(float(speed), 0.1)

    def StartSimulation(
        self,
        skip_summary: bool = False,
        delete_results: bool = False,
        non_blocking: bool = False,
    ) -> bool:
        """Start the simulation."""
        if self._thread is not None and self._thread.is_alive():
            return False
        if delete_results:
            self.DeleteResults()
        self._elapsed = 0.0
        self._progress = 0.0
        self._has_results = False
        self._simulating = True
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run_simulation, daemon=True)
        self._thread.start()
        if not non_blocking:
            self._thread.join()
        return True

    def _run_simulation(self) -> None:
        """Advance the simulated clock."""
        duration = self._solver.GetSimulationDuration()
        wall_seconds = max(0.2, duration / max(self._speed, 0.1))
        steps = 40
        for index in range(steps):
            if self._stop_event.is_set():
                break
            self._stop_event.wait(wall_seconds / steps)
            self._elapsed = duration * (index + 1) / steps
            self._progress = 100.0 * (index + 1) / steps
        if not self._stop_event.is_set():
            self._elapsed = duration
            self._progress = 100.0
            self._has_results = True
        self._simulating = False

    def StopSimulation(self) -> None:
        """Stop a running simulation."""
        self._stop_event.set()

    def IsSimulating(self) -> bool:
        """Return whether a simulation is running."""
        return self._simulating

    def GetProgress(self) -> float | None:
        """Return the simulation progress in percent."""
        return self._progress

    def HasResults(self) -> bool:
        """Return whether results are available."""
        return self._has_results

    def RefreshResults(self) -> None:
        """Refresh the available results."""
        if self._elapsed > 0:
            self._has_results = True

    def DeleteResults(self) -> None:
        """Delete the computed results."""
        self._elapsed = 0.0
        self._progress = 0.0
        self._has_results = False

    def ResetSimulationStatus(self) -> None:
        """Reset the simulation status."""
        self._progress = 0.0

    def GetTimeSet(self) -> list[float]:
        """Return the computed time set."""
        if self._has_results or self._simulating:
            return self._computed_time_set()
        return self._planned_time_set()

    def UpdateTimeSet(self) -> None:
        """Update the time set (no-op in the mock)."""

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def _planned_time_set(self) -> list[float]:
        """Return the configured time set."""
        duration = self._solver.GetSimulationDuration()
        interval = self._solver.GetTimeInterval() or max(duration / 20.0, 1e-9)
        count = max(2, int(round(duration / interval)) + 1)
        return [duration * index / (count - 1) for index in range(count)]

    def _computed_time_set(self) -> list[float]:
        """Return the time set truncated at the current simulated time."""
        elapsed = self._elapsed
        return [time for time in self._planned_time_set() if time <= elapsed + 1e-9] or [0.0]

    def _resolve_time(self, time_step: Any) -> float:
        """Resolve a time-step reference into a simulation time in seconds."""
        times = self._computed_time_set() if self._has_results else self._planned_time_set()
        if time_step is None or time_step == "current":
            return float(times[-1])
        if isinstance(time_step, (int, float)) and not isinstance(time_step, bool):
            index = int(time_step)
            if index < 0:
                index += len(times)
            if 0 <= index < len(times):
                return float(times[index])
            return float(times[-1])
        return float(time_step)

    def _derived(self) -> dict[str, float]:
        """Return the derived flow quantities of the study."""
        mass_flow_kg_s = 0.0
        inlet_radius = DEFAULT_INLET_RADIUS
        for inlet in self._inlets:
            properties = inlet.GetInputPropertiesList()[0]
            mass_flow_kg_s += properties.GetMassFlowRate("kg/s")
            surface = inlet.GetEntryPoint()
            if isinstance(surface, MockSurface):
                inlet_radius = max(surface.GetMaxRadius("m"), 1e-6)
                break

        area = math.pi * inlet_radius**2
        volumetric_flow = mass_flow_kg_s / BULK_DENSITY if mass_flow_kg_s else 0.0
        velocity = volumetric_flow / area if area > 0 else 0.0

        sieve = DEFAULT_SIEVE_SIZE
        if self._particles:
            sieve = max(self._particles[0].GetMinimumSize("m"), 1e-6)
        particle_mass = PARTICLE_DENSITY * (4.0 / 3.0) * math.pi * (sieve / 2.0) ** 3
        rate = mass_flow_kg_s / particle_mass if particle_mass > 0 else 0.0

        min_corner, max_corner = self._domain.GetCoordinateLimitsMinMax()
        domain_length = max(max_corner[0] - min_corner[0], 1e-6)
        transit = domain_length / velocity if velocity > 0 else math.inf

        return {
            "mass_flow_kg_s": mass_flow_kg_s,
            "velocity": velocity,
            "rate": rate,
            "transit": transit,
            "domain_length": domain_length,
        }

    def _particle_count(self, time: float) -> float:
        """Return the number of particles inside the domain at a given time."""
        derived = self._derived()
        transit = derived["transit"]
        held = min(time, transit) if transit != math.inf else time
        return derived["rate"] * held

    def _curve_data(
        self, curve_name: str
    ) -> tuple[list[float], list[float], str | None, str | None]:
        """Return ``(times, values, base_unit, default_unit)`` for a named curve.

        The values are stored in the base unit of the curve family
        (``kg/s`` for mass flow, ``m/s`` for velocities) and converted to the
        requested unit by the caller.
        """
        if not self._has_results:
            raise RuntimeError(
                "No results are available yet. Run 'rocky_run_simulation' before requesting curves."
            )
        times = self._computed_time_set()
        derived = self._derived()
        transit = derived["transit"]
        mass_flow_kg_s = derived["mass_flow_kg_s"]

        if curve_name in ("Particles Count", "Particle Count"):
            values = [self._particle_count(time) for time in times]
            base_unit = None
            default_unit = None
        elif curve_name in ("Particles Mass Flow In", "Mass Flow In"):
            values = [mass_flow_kg_s if time > 0 else 0.0 for time in times]
            base_unit = "kg/s"
            default_unit = "t/h"
        elif curve_name in ("Particles Mass Flow Out", "Mass Flow Out"):
            values = []
            for time in times:
                if transit == math.inf or time <= transit:
                    values.append(0.0)
                else:
                    ramp = min(1.0, (time - transit) / max(0.2 * transit, 1e-9))
                    values.append(mass_flow_kg_s * ramp)
            base_unit = "kg/s"
            default_unit = "t/h"
        elif curve_name == "Total Mass":
            values = [mass_flow_kg_s * min(time, transit) for time in times]
            base_unit = "kg/s"
            default_unit = "t/h"
        elif curve_name == "Kinetic Energy":
            velocity = derived["velocity"]
            values = [0.5 * mass_flow_kg_s * min(time, transit) * velocity**2 for time in times]
            base_unit = None
            default_unit = None
        else:
            available = (
                "Particles Count",
                "Particles Mass Flow In",
                "Particles Mass Flow Out",
                "Total Mass",
                "Kinetic Energy",
            )
            raise KeyError(
                f"Unknown curve {curve_name!r}. Available curves: {', '.join(available)}."
            )
        return times, values, base_unit, default_unit

    def _remove_selection(self, process: MockSelectionProcess) -> None:
        """Remove a selection process from the study."""
        if process in self._selections:
            self._selections.remove(process)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the study."""
        derived = self._derived()
        return {
            "name": self._name,
            "particles": [
                {
                    "name": particle.GetName(),
                    "shape": particle.GetShape(),
                    "sieve_size_m": particle.GetMinimumSize("m"),
                }
                for particle in self._particles
            ],
            "surfaces": [
                {
                    "name": surface.GetName(),
                    "kind": surface.GetKind(),
                    "center_m": surface.GetCenter("m"),
                    "max_radius_m": surface.GetMaxRadius("m"),
                    "length_m": surface.GetLength("m"),
                    "width_m": surface.GetWidth("m"),
                }
                for surface in self._surfaces
            ],
            "inlets": [
                {
                    "name": inlet.GetName(),
                    "entry_point": inlet.GetEntryPoint().GetName(),
                    "mass_flow_t_h": inlet.GetInputPropertiesList()[0].GetMassFlowRate("t/h"),
                }
                for inlet in self._inlets
            ],
            "outlets": [
                {"name": outlet.GetName(), "exit_point": outlet.GetExitPoint().GetName()}
                for outlet in self._outlets
            ],
            "walls": [
                {"name": wall.GetName(), "filename": wall.GetFilename()} for wall in self._walls
            ],
            "selections": [
                {
                    "name": selection.GetName(),
                    "kind": type(selection).__name__.replace("Mock", "").replace("Process", ""),
                    "size_m": selection.GetSize("m"),
                    "center_m": selection.GetCenter("m"),
                }
                for selection in self._selections
            ],
            "domain": {
                "min_m": self._domain.GetCoordinateLimitsMinValues("m"),
                "max_m": self._domain.GetCoordinateLimitsMaxValues("m"),
                "use_boundary_limits": self._domain.GetUseBoundaryLimits(),
            },
            "solver": {
                "duration_s": self._solver.GetSimulationDuration(),
                "time_interval_s": self._solver.GetTimeInterval(),
                "processors": self._solver.GetNumberOfProcessors(),
                "execution_mode": self._solver.GetExecutionMode(),
            },
            "derived": {
                "inlet_velocity_m_s": derived["velocity"],
                "particle_rate_per_s": derived["rate"],
                "transit_time_s": (None if derived["transit"] == math.inf else derived["transit"]),
            },
            "simulation": {
                "simulating": self.IsSimulating(),
                "progress_percent": self.GetProgress(),
                "has_results": self.HasResults(),
            },
        }


class MockProject:
    """Stand-in for a Rocky project."""

    def __init__(self, filename: str | None = None) -> None:
        """Initialize the project."""
        self._filename = filename
        self._studies: dict[str, MockStudy] = {}
        self._processes: MockUserProcessCollection | None = None
        self._unsaved = False

    def GetStudy(self, study_name: str | None = None) -> MockStudy:
        """Return a study, creating the default one on first use."""
        if study_name and study_name in self._studies:
            return self._studies[study_name]
        if not self._studies:
            self._studies["Study 1"] = MockStudy("Study 1")
        return next(iter(self._studies.values()))

    def CreateStudy(self, study_name: str | None = None) -> MockStudy:
        """Create a named study."""
        name = study_name or f"Study {len(self._studies) + 1}"
        study = MockStudy(name)
        self._studies[name] = study
        return study

    def GetStudyNames(self) -> list[str]:
        """Return the study names."""
        return list(self._studies)

    def GetUserProcessCollection(self) -> MockUserProcessCollection:
        """Return the user process collection of the first study."""
        return self.GetStudy().GetUserProcessCollection()

    def GetProjectFilename(self) -> str:
        """Return the path of the saved project file."""
        return self._filename or ""

    def HasUnsavedChanges(self) -> bool:
        """Return whether the project has unsaved changes."""
        return self._unsaved

    def SaveProject(self, filename: str | None = None) -> None:
        """Save the project to disk."""
        target = Path(filename or self._filename or "project.rocky")
        if target.parent and str(target.parent) not in ("", "."):
            target.parent.mkdir(parents=True, exist_ok=True)
        if not self._studies:
            self.GetStudy()
        payload = {
            "format": "ansys-rocky-mcp-mock",
            "version": 1,
            "project": {
                "studies": {name: study.to_dict() for name, study in self._studies.items()}
            },
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self._filename = str(target)
        self._unsaved = False

    def CloseProject(self, check_save_state: bool = True) -> None:
        """Close the project."""


class MockRockyApplication:
    """Stand-in for the Rocky API application object."""

    def __init__(self, version: str = "2026R1 (mock)") -> None:
        """Initialize the application."""
        self._version = version
        self._project: MockProject | None = None

    def GetVersion(self) -> str:
        """Return the Rocky version."""
        return self._version

    def CreateProject(self) -> MockProject:
        """Create a new project."""
        self._project = MockProject()
        return self._project

    def GetProject(self) -> MockProject | None:
        """Return the current project, if any."""
        return self._project

    def OpenProject(self, filename: str) -> MockProject:
        """Open a project from disk."""
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Rocky project file not found: {filename}")
        self._project = MockProject(str(path))
        return self._project

    def SaveProject(self, filename: str | None = None) -> None:
        """Save the current project."""
        if self._project is None:
            raise RuntimeError("There is no open project to save.")
        self._project.SaveProject(filename)

    def CloseProject(self, check_save_state: bool = True) -> None:
        """Close the current project."""
        self._project = None

    def GetStudy(self, study_name: str | None = None) -> MockStudy:
        """Return a study of the current project."""
        if self._project is None:
            raise RuntimeError("There is no open project.")
        return self._project.GetStudy(study_name)

    def Exit(self) -> None:
        """Exit the application."""
        self._project = None


class MockBackend(RockyBackend):
    """Backend that serves the in-memory Rocky simulator."""

    mode = "mock"
    simulated = True

    def __init__(self, speed: float = 8.0) -> None:
        """Initialize the backend."""
        self._app = MockRockyApplication()
        self._speed = speed
        self._app.GetStudy  # noqa: B018 - keep the attribute discoverable for linters

    def probe(self) -> BackendStatus:
        """Report the mock backend status."""
        return BackendStatus(
            mode=self.mode,
            connected=True,
            simulated=True,
            rocky_available=False,
            rocky_version="2026R1 (mock)",
            installed_versions=[],
            detail=(
                "Running against the built-in Rocky simulator. Results are physically "
                "consistent but SIMULATED, not computed by Ansys Rocky."
            ),
        )

    def connect(self, mode: str = "auto") -> MockRockyApplication:
        """Return the mock application object."""
        return self._app

    def close(self) -> None:
        """Reset the mock application."""
        self._app = MockRockyApplication()
        self._app.GetStudy  # noqa: B018

    @property
    def app(self) -> MockRockyApplication:
        """Return the mock application object."""
        return self._app

    def apply_speed(self, study: MockStudy) -> None:
        """Propagate the configured mock speed to a study."""
        study.SetMockSpeed(self._speed)
