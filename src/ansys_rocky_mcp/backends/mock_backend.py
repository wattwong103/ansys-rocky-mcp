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


def _require_kelvin(unit: str | None) -> None:
    """Reject a temperature unit other than Rocky's ``K``."""
    if unit not in (None, "K"):
        raise ValueError(f"Unsupported temperature unit {unit!r}. Expected 'K'.")


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

    def GetCurveNames(self) -> list[str]:
        """Return the curve names this process can read."""
        return [
            "Particles Count",
            "Particles Mass Flow In",
            "Particles Mass Flow Out",
            "Total Mass",
            "Kinetic Energy",
        ]

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


class MockSolidMaterial:
    """Stand-in for a Rocky solid material."""

    def __init__(self, name: str) -> None:
        """Initialize the material."""
        self._name = name
        self._density = 1000.0
        self._youngs = 1.0e8
        self._poisson = 0.25
        self._specific_heat = 1000.0
        self._conductivity = 1.0

    def GetName(self) -> str:
        """Return the material name."""
        return self._name

    def SetDensity(self, value: float, unit: str | None = None) -> None:
        """Set the solid density."""
        self._density = float(value)

    def GetDensity(self, unit: str | None = None) -> float:
        """Return the solid density."""
        return self._density

    def SetYoungsModulus(self, value: float, unit: str | None = None) -> None:
        """Set Young's modulus. Rocky's token is ``N/m2``."""
        if unit not in (None, "N/m2"):
            raise ValueError(f"Unsupported modulus unit {unit!r}. Expected 'N/m2'.")
        self._youngs = float(value)

    def GetYoungsModulus(self, unit: str | None = None) -> float:
        """Return Young's modulus in N/m2."""
        if unit not in (None, "N/m2"):
            raise ValueError(f"Unsupported modulus unit {unit!r}. Expected 'N/m2'.")
        return self._youngs

    def SetSpecificHeat(self, value: float, unit: str | None = None) -> None:
        """Set specific heat. Rocky's token is ``J/kg.K``."""
        if unit not in (None, "J/kg.K"):
            raise ValueError(f"Unsupported specific heat unit {unit!r}. Expected 'J/kg.K'.")
        self._specific_heat = float(value)

    def GetSpecificHeat(self, unit: str | None = None) -> float:
        """Return specific heat in J/kg.K."""
        if unit not in (None, "J/kg.K"):
            raise ValueError(f"Unsupported specific heat unit {unit!r}. Expected 'J/kg.K'.")
        return self._specific_heat

    def SetThermalConductivity(self, value: float, unit: str | None = None) -> None:
        """Set thermal conductivity. Rocky's token is ``W/m.K``."""
        if unit not in (None, "W/m.K"):
            raise ValueError(f"Unsupported conductivity unit {unit!r}. Expected 'W/m.K'.")
        self._conductivity = float(value)

    def GetThermalConductivity(self, unit: str | None = None) -> float:
        """Return thermal conductivity in W/m.K."""
        if unit not in (None, "W/m.K"):
            raise ValueError(f"Unsupported conductivity unit {unit!r}. Expected 'W/m.K'.")
        return self._conductivity

    def SetPoissonRatio(self, value: float, unit: str | None = None) -> None:
        """Set Poisson's ratio."""
        self._poisson = float(value)

    def GetPoissonRatio(self, unit: str | None = None) -> float:
        """Return Poisson's ratio."""
        return self._poisson


class MockMaterialsInteraction:
    """Stand-in for the contact pair between two materials."""

    def __init__(self) -> None:
        """Initialize neutral contact coefficients."""
        self._static = 0.0
        self._dynamic = 0.0
        self._restitution = 0.0
        self._adhesive_distance = 0.0
        self._stiffness_ratio = 0.0
        self._velocity_exponent = 0.0
        self._velocity_limit = 0.0
        self._stiffness_multiplier = 1.0
        self._adhesive_fraction = 0.0
        self._pair_surface_energy = 0.0

    def SetStaticFriction(self, value: float, unit: str | None = None) -> None:
        """Set the static friction coefficient."""
        self._static = float(value)

    def GetStaticFriction(self, unit: str | None = None) -> float:
        """Return the static friction coefficient."""
        return self._static

    def SetDynamicFriction(self, value: float, unit: str | None = None) -> None:
        """Set the dynamic friction coefficient."""
        self._dynamic = float(value)

    def GetDynamicFriction(self, unit: str | None = None) -> float:
        """Return the dynamic friction coefficient."""
        return self._dynamic

    def SetRestitutionCoefficient(self, value: float, unit: str | None = None) -> None:
        """Set the restitution coefficient."""
        self._restitution = float(value)

    def GetRestitutionCoefficient(self, unit: str | None = None) -> float:
        """Return the restitution coefficient."""
        return self._restitution

    def SetAdhesiveDistance(self, value: float, unit: str | None = None) -> None:
        """Set the adhesive distance."""
        self._adhesive_distance = float(value)

    def GetAdhesiveDistance(self, unit: str | None = None) -> float:
        """Return the adhesive distance."""
        return self._adhesive_distance

    def SetTangentialStiffnessRatio(self, value: float, unit: str | None = None) -> None:
        """Set the tangential stiffness ratio."""
        self._stiffness_ratio = float(value)

    def GetTangentialStiffnessRatio(self, unit: str | None = None) -> float:
        """Return the tangential stiffness ratio."""
        return self._stiffness_ratio

    def SetVelocityExponent(self, value: float, unit: str | None = None) -> None:
        """Set the velocity exponent."""
        self._velocity_exponent = float(value)

    def GetVelocityExponent(self, unit: str | None = None) -> float:
        """Return the velocity exponent."""
        return self._velocity_exponent

    def SetVelocityLimit(self, value: float, unit: str | None = None) -> None:
        """Set the velocity limit."""
        self._velocity_limit = float(value)

    def GetVelocityLimit(self, unit: str | None = None) -> float:
        """Return the velocity limit."""
        return self._velocity_limit

    def SetContactStiffnessMultiplier(self, value: float, unit: str | None = None) -> None:
        """Set the contact stiffness multiplier."""
        self._stiffness_multiplier = float(value)

    def GetContactStiffnessMultiplier(self, unit: str | None = None) -> float:
        """Return the contact stiffness multiplier."""
        return self._stiffness_multiplier

    def SetAdhesiveFraction(self, value: float, unit: str | None = None) -> None:
        """Set the adhesive fraction."""
        self._adhesive_fraction = float(value)

    def GetAdhesiveFraction(self, unit: str | None = None) -> float:
        """Return the adhesive fraction."""
        return self._adhesive_fraction

    def SetSurfaceEnergy(self, value: float, unit: str | None = None) -> None:
        """Set the pair surface energy."""
        self._pair_surface_energy = float(value)

    def GetSurfaceEnergy(self, unit: str | None = None) -> float:
        """Return the pair surface energy."""
        return self._pair_surface_energy


class MockMaterialsInteractionCollection:
    """Stand-in for the study material-interaction collection."""

    def __init__(self) -> None:
        """Initialize an empty pair map."""
        self._pairs: dict[tuple[int, int], MockMaterialsInteraction] = {}

    def GetMaterialsInteraction(self, material_1: Any, material_2: Any) -> MockMaterialsInteraction:
        """Return the interaction for a material pair, creating it if needed."""
        key = tuple(sorted((id(material_1), id(material_2))))
        interaction = self._pairs.get(key)
        if interaction is None:
            interaction = MockMaterialsInteraction()
            self._pairs[key] = interaction
        return interaction


class MockMaterialCollection:
    """Stand-in for the study material collection."""

    def __init__(self) -> None:
        """Initialize an empty collection."""
        self._materials: list[MockSolidMaterial] = []

    def AddSolidMaterial(self, name: str | None = None) -> MockSolidMaterial:
        """Create a solid material."""
        material = MockSolidMaterial(name or f"Material {len(self._materials) + 1}")
        self._materials.append(material)
        return material

    def GetSolidMaterial(self, material_name: str) -> MockSolidMaterial:
        """Return a solid material by name."""
        for material in self._materials:
            if material.GetName() == material_name:
                return material
        raise KeyError(material_name)


class MockParticle:
    """Stand-in for a Rocky particle definition."""

    def __init__(self, name: str = "Particle") -> None:
        """Initialize the particle."""
        self._name = name
        self._shape = "sphere"
        self._size_type = "sieve"
        self._minimum_size = DEFAULT_SIEVE_SIZE
        self._distribution = "gaudin_schumann"
        self._size_ratio = 1.0
        self._breakage_model = "none"
        self._breakage = False
        self._abt10_size = 0.0
        self._abt10_energy = 0.0
        self._tavares_a = 0.0
        self._tavares_d0 = 0.0
        self._tavares_e_inf = 0.0
        self._shear_limit = 0.0
        self._tensile_limit = 0.0
        self._von_mises_limit = 0.0
        self._plastic_limit = 0.0
        self._porosity = 0.0
        self._bending_deg = 0.0
        self._torsion_deg = 0.0
        self._flexible = False
        self._elasticity = "isotropic"
        self._target_elements = 1
        self._ratio_bending = 1.0
        self._ratio_normal = 1.0
        self._ratio_tangential = 1.0
        self._ratio_torsion = 1.0
        self._element_damping = 0.0
        self._joint_damping = 0.0
        self._with_failure = False
        self._failure_ratio = 1.0
        self._random_angle = False
        self._random_half_rad = (0.0, 0.0, 0.0)
        self._surface_energy = 0.0
        self._material: MockSolidMaterial | None = None
        self._horizontal_aspect = 1.0
        self._vertical_aspect = 1.0
        self._thickness = 0.0
        self._corners = 4
        self._edge_deg = 0.0
        self._rolling_resistance = 0.0

    def GetName(self) -> str:
        """Return the particle name."""
        return self._name

    def SetMaterial(self, value: Any) -> None:
        """Assign a solid material by object or name."""
        self._material = value

    def GetMaterial(self) -> Any:
        """Return the assigned material."""
        return self._material

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

    def GetHorizontalAspectRatio(self) -> float:
        """Return the horizontal aspect ratio."""
        return self._horizontal_aspect

    def SetHorizontalAspectRatio(self, value: float) -> None:
        """Set the horizontal aspect ratio."""
        self._horizontal_aspect = float(value)

    def GetVerticalAspectRatio(self) -> float:
        """Return the vertical aspect ratio."""
        return self._vertical_aspect

    def SetVerticalAspectRatio(self, value: float) -> None:
        """Set the vertical aspect ratio."""
        self._vertical_aspect = float(value)

    def GetThickness(self, unit: str | None = None) -> float:
        """Return the particle thickness in meters."""
        if unit not in (None, "m"):
            raise ValueError(f"Unsupported length unit {unit!r}. Expected 'm'.")
        return self._thickness

    def SetThickness(self, value: float, unit: str | None = None) -> None:
        """Set the particle thickness in meters."""
        if unit not in (None, "m"):
            raise ValueError(f"Unsupported length unit {unit!r}. Expected 'm'.")
        self._thickness = float(value)

    def GetNumberOfCorners(self) -> int:
        """Return the number of corners."""
        return self._corners

    def SetNumberOfCorners(self, value: int) -> None:
        """Set the number of corners."""
        self._corners = int(value)

    def GetEdgeAngle(self, unit: str | None = None) -> float:
        """Return the edge angle in degrees."""
        if unit not in (None, "dega"):
            raise ValueError(f"Unsupported angle unit {unit!r}. Expected 'dega'.")
        return self._edge_deg

    def SetEdgeAngle(self, value: float, unit: str | None = None) -> None:
        """Set the edge angle. Rocky's degree token is ``dega``."""
        if unit not in (None, "dega"):
            raise ValueError(f"Unsupported angle unit {unit!r}. Expected 'dega'.")
        self._edge_deg = float(value)

    def GetSizeType(self) -> str:
        """Return the particle size type."""
        return self._size_type

    def SetSizeType(self, value: str) -> None:
        """Set the particle size type."""
        valid = ("sieve", "equivalent_diameter", "original_size_scale")
        if value not in valid:
            raise ValueError(f"Unknown size type {value!r}. Expected one of {valid}.")
        self._size_type = value

    def GetRollingResistance(self) -> float:
        """Return the particle rolling resistance."""
        return self._rolling_resistance

    def SetRollingResistance(self, value: float) -> None:
        """Set the particle rolling resistance."""
        self._rolling_resistance = float(value)

    def GetMinimumSize(self, unit: str | None = None) -> float:
        """Return the sieve size in the requested unit."""
        factor = 1.0 / _LENGTH_TO_M.get(unit or "m", 1.0)
        return self._minimum_size * factor

    def SetMinimumSize(self, value: float, unit: str | None = None) -> None:
        """Set the sieve size."""
        self._minimum_size = _length_to_m(value, unit)

    def GetDistributionModel(self) -> str:
        """Return the size distribution model."""
        return self._distribution

    def SetDistributionModel(self, value: str) -> None:
        """Set the size distribution model."""
        self._distribution = value

    def GetMinimumSizeRatio(self) -> float:
        """Return the minimum size as a fraction of the maximum size."""
        return self._size_ratio

    def SetMinimumSizeRatio(self, value: float) -> None:
        """Set the minimum size as a fraction of the maximum size."""
        self._size_ratio = float(value)

    def GetBreakageModel(self) -> str:
        """Return the breakage model."""
        return self._breakage_model

    def SetBreakageModel(self, value: str) -> None:
        """Set the breakage model. Any model other than none enables breakage."""
        self._breakage_model = value
        if value != "none":
            self._breakage = True

    def GetEnableBreakage(self) -> bool:
        """Return whether breakage is enabled."""
        return self._breakage

    def SetEnableBreakage(self, value: bool) -> None:
        """Enable or disable breakage."""
        self._breakage = bool(value)

    def GetAbt10ReferenceSize(self, unit: str | None = None) -> float:
        """Return the Abt10 reference size in meters."""
        if unit not in (None, "m"):
            raise ValueError(f"Unsupported length unit {unit!r}. Expected 'm'.")
        return self._abt10_size

    def SetAbt10ReferenceSize(self, value: float, unit: str | None = None) -> None:
        """Set the Abt10 reference size in meters."""
        if unit not in (None, "m"):
            raise ValueError(f"Unsupported length unit {unit!r}. Expected 'm'.")
        self._abt10_size = float(value)

    def GetAbt10ReferenceMinimumSpecificEnergy(self, unit: str | None = None) -> float:
        """Return the Abt10 reference energy in joules per kilogram."""
        if unit not in (None, "J/kg"):
            raise ValueError(f"Unsupported energy unit {unit!r}. Expected 'J/kg'.")
        return self._abt10_energy

    def SetAbt10ReferenceMinimumSpecificEnergy(self, value: float, unit: str | None = None) -> None:
        """Set the Abt10 reference energy in joules per kilogram."""
        if unit not in (None, "J/kg"):
            raise ValueError(f"Unsupported energy unit {unit!r}. Expected 'J/kg'.")
        self._abt10_energy = float(value)

    def GetTavaresA(self, unit: str | None = None) -> float:
        """Return Tavares A as a percent."""
        if unit not in (None, "%"):
            raise ValueError(f"Unsupported percent unit {unit!r}. Expected '%'.")
        return self._tavares_a

    def SetTavaresA(self, value: float, unit: str | None = None) -> None:
        """Set Tavares A. Rocky's token is percent."""
        if unit not in (None, "%"):
            raise ValueError(f"Unsupported percent unit {unit!r}. Expected '%'.")
        self._tavares_a = float(value)

    def GetTavaresD0(self, unit: str | None = None) -> float:
        """Return Tavares D0 in meters."""
        if unit not in (None, "m"):
            raise ValueError(f"Unsupported length unit {unit!r}. Expected 'm'.")
        return self._tavares_d0

    def SetTavaresD0(self, value: float, unit: str | None = None) -> None:
        """Set Tavares D0 in meters."""
        if unit not in (None, "m"):
            raise ValueError(f"Unsupported length unit {unit!r}. Expected 'm'.")
        self._tavares_d0 = float(value)

    def GetTavaresEInf(self, unit: str | None = None) -> float:
        """Return Tavares E infinity in joules per kilogram."""
        if unit not in (None, "J/kg"):
            raise ValueError(f"Unsupported energy unit {unit!r}. Expected 'J/kg'.")
        return self._tavares_e_inf

    def SetTavaresEInf(self, value: float, unit: str | None = None) -> None:
        """Set Tavares E infinity in joules per kilogram."""
        if unit not in (None, "J/kg"):
            raise ValueError(f"Unsupported energy unit {unit!r}. Expected 'J/kg'.")
        self._tavares_e_inf = float(value)

    def _stress(self, unit: str | None) -> None:
        """Reject a stress unit other than newtons per square meter."""
        if unit not in (None, "N/m2"):
            raise ValueError(f"Unsupported stress unit {unit!r}. Expected 'N/m2'.")

    def GetShearStressLimit(self, unit: str | None = None) -> float:
        """Return the shear stress limit in N/m2."""
        self._stress(unit)
        return self._shear_limit

    def SetShearStressLimit(self, value: float, unit: str | None = None) -> None:
        """Set the shear stress limit in N/m2."""
        self._stress(unit)
        self._shear_limit = float(value)

    def GetTensileStressLimit(self, unit: str | None = None) -> float:
        """Return the tensile stress limit in N/m2."""
        self._stress(unit)
        return self._tensile_limit

    def SetTensileStressLimit(self, value: float, unit: str | None = None) -> None:
        """Set the tensile stress limit in N/m2."""
        self._stress(unit)
        self._tensile_limit = float(value)

    def GetVonMisesStressLimit(self, unit: str | None = None) -> float:
        """Return the von Mises stress limit in N/m2."""
        self._stress(unit)
        return self._von_mises_limit

    def SetVonMisesStressLimit(self, value: float, unit: str | None = None) -> None:
        """Set the von Mises stress limit in N/m2."""
        self._stress(unit)
        self._von_mises_limit = float(value)

    def GetPlasticLimit(self, unit: str | None = None) -> float:
        """Return the plastic limit in pascals. Rocky's token is ``Pa``."""
        if unit not in (None, "Pa"):
            raise ValueError(f"Unsupported plastic-limit unit {unit!r}. Expected 'Pa'.")
        return self._plastic_limit

    def SetPlasticLimit(self, value: float, unit: str | None = None) -> None:
        """Set the plastic limit. Rocky's token is ``Pa``, not ``N/m2``."""
        if unit not in (None, "Pa"):
            raise ValueError(f"Unsupported plastic-limit unit {unit!r}. Expected 'Pa'.")
        self._plastic_limit = float(value)

    def GetPorosity(self, unit: str | None = None) -> float:
        """Return porosity. Rocky's token is ``<fraction>``."""
        if unit not in (None, "<fraction>"):
            raise ValueError(f"Unsupported porosity unit {unit!r}. Expected '<fraction>'.")
        return self._porosity

    def SetPorosity(self, value: float, unit: str | None = None) -> None:
        """Set porosity. Rocky's token is ``<fraction>``."""
        if unit not in (None, "<fraction>"):
            raise ValueError(f"Unsupported porosity unit {unit!r}. Expected '<fraction>'.")
        self._porosity = float(value)

    def GetBendingAngleLimit(self, unit: str | None = None) -> float:
        """Return the bending angle limit in degrees."""
        if unit not in (None, "dega"):
            raise ValueError(f"Unsupported angle unit {unit!r}. Expected 'dega'.")
        return self._bending_deg

    def SetBendingAngleLimit(self, value: float, unit: str | None = None) -> None:
        """Set the bending angle limit. Rocky's degree token is ``dega``."""
        if unit not in (None, "dega"):
            raise ValueError(f"Unsupported angle unit {unit!r}. Expected 'dega'.")
        self._bending_deg = float(value)

    def GetTorsionAngleLimit(self, unit: str | None = None) -> float:
        """Return the torsion angle limit in degrees."""
        if unit not in (None, "dega"):
            raise ValueError(f"Unsupported angle unit {unit!r}. Expected 'dega'.")
        return self._torsion_deg

    def SetTorsionAngleLimit(self, value: float, unit: str | None = None) -> None:
        """Set the torsion angle limit. Rocky's degree token is ``dega``."""
        if unit not in (None, "dega"):
            raise ValueError(f"Unsupported angle unit {unit!r}. Expected 'dega'.")
        self._torsion_deg = float(value)

    def GetFlexible(self) -> bool:
        """Return whether the particle is flexible."""
        return self._flexible

    def SetFlexible(self, value: bool) -> None:
        """Set whether the particle is flexible."""
        self._flexible = bool(value)

    def GetElasticity(self) -> str:
        """Return the elasticity model."""
        return self._elasticity

    def SetElasticity(self, value: str) -> None:
        """Set the elasticity model."""
        if value not in ("isotropic", "anisotropic"):
            raise ValueError(f"Unknown elasticity {value!r}.")
        self._elasticity = value

    def GetTargetNumberOfElements(self) -> int:
        """Return the target number of fiber elements."""
        return self._target_elements

    def SetTargetNumberOfElements(self, value: int) -> None:
        """Set the target number of fiber elements."""
        count = int(value)
        if count < 1:
            raise ValueError("Target number of elements must be at least 1.")
        self._target_elements = count

    def GetElasticRatioBending(self) -> float:
        """Return the bending stiffness ratio."""
        return self._ratio_bending

    def SetElasticRatioBending(self, value: float) -> None:
        """Set the bending stiffness ratio."""
        self._ratio_bending = float(value)

    def GetElasticRatioNormal(self) -> float:
        """Return the normal stiffness ratio."""
        return self._ratio_normal

    def SetElasticRatioNormal(self, value: float) -> None:
        """Set the normal stiffness ratio."""
        self._ratio_normal = float(value)

    def GetElasticRatioTangential(self) -> float:
        """Return the tangential stiffness ratio."""
        return self._ratio_tangential

    def SetElasticRatioTangential(self, value: float) -> None:
        """Set the tangential stiffness ratio."""
        self._ratio_tangential = float(value)

    def GetElasticRatioTorsion(self) -> float:
        """Return the torsion stiffness ratio."""
        return self._ratio_torsion

    def SetElasticRatioTorsion(self, value: float) -> None:
        """Set the torsion stiffness ratio."""
        self._ratio_torsion = float(value)

    def GetElementDampingRatio(self) -> float:
        """Return the element damping ratio."""
        return self._element_damping

    def SetElementDampingRatio(self, value: float) -> None:
        """Set the element damping ratio."""
        self._element_damping = float(value)

    def GetJointDampingRatio(self) -> float:
        """Return the joint damping ratio."""
        return self._joint_damping

    def SetJointDampingRatio(self, value: float) -> None:
        """Set the joint damping ratio."""
        self._joint_damping = float(value)

    def GetWithFailure(self) -> bool:
        """Return whether the fiber can fail."""
        return self._with_failure

    def SetWithFailure(self, value: bool) -> None:
        """Set whether the fiber can fail."""
        self._with_failure = bool(value)

    def GetFailureRatio(self) -> float:
        """Return the failure ratio."""
        return self._failure_ratio

    def SetFailureRatio(self, value: float) -> None:
        """Set the failure ratio."""
        self._failure_ratio = float(value)

    def GetEnableRandomAngle(self) -> bool:
        """Return whether the particle is given a random orientation."""
        return self._random_angle

    def SetEnableRandomAngle(self, value: bool) -> None:
        """Set whether the particle is given a random orientation."""
        self._random_angle = bool(value)

    def GetRandomAnglesHalfRange(self, unit: str | None = None) -> list[float]:
        """Return the random-angle half ranges in radians, or degrees when asked."""
        if unit in ("deg", "dega"):
            return [math.degrees(value) for value in self._random_half_rad]
        if unit not in (None, "rad"):
            raise ValueError(f"Unsupported angle unit {unit!r}.")
        return list(self._random_half_rad)

    def SetRandomAnglesHalfRange(self, values: list[float], unit: str | None = None) -> None:
        """Set the random-angle half ranges. Degrees use Rocky's ``dega`` token."""
        if len(values) != 3:
            raise ValueError("Random angle half range must have three numbers.")
        if unit in ("deg", "dega"):
            self._random_half_rad = tuple(math.radians(float(value)) for value in values)
        elif unit in (None, "rad"):
            self._random_half_rad = tuple(float(value) for value in values)
        else:
            raise ValueError(f"Unsupported angle unit {unit!r}.")

    def GetSurfaceEnergy(self, unit: str | None = None) -> float:
        """Return the surface energy."""
        return self._surface_energy

    def SetSurfaceEnergy(self, value: float, unit: str | None = None) -> None:
        """Set the surface energy."""
        self._surface_energy = float(value)


class MockParticleInletProperties:
    """Stand-in for a particle inlet property set."""

    def __init__(self, particle: MockParticle) -> None:
        """Initialize the properties for a particle."""
        self._particle = particle
        self._mass_flow_tph = 0.0
        self._temperature_k = 298.15

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

    def GetTemperature(self, unit: str | None = None) -> float:
        """Return the inlet temperature in kelvin."""
        _require_kelvin(unit)
        return self._temperature_k

    def SetTemperature(self, value: float, unit: str | None = None) -> None:
        """Set the inlet temperature in kelvin."""
        _require_kelvin(unit)
        self._temperature_k = float(value)


class MockParticleInlet:
    """Stand-in for a Rocky particle inlet."""

    def __init__(self, entry_point: Any, particle: MockParticle) -> None:
        """Initialize the inlet."""
        self._entry_point = entry_point
        self._particle = particle
        self._name = "Particle Inlet"
        self._properties = [MockParticleInletProperties(particle)]
        self._injection_duration = 0.0
        self._start_time = 0.0
        self._stop_time = 0.0
        self._target_velocity = 0.0
        self._use_target_velocity = False
        self._periodic = False
        self._period = 0.0
        self._ux = 0.0
        self._uz = 0.0
        self._stop_all = False
        self._force_packing = False
        self._sph_temperature = 298.15
        self._sph_injection = False

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

    def AddParticle(self, particle: MockParticle) -> list[MockParticleInletProperties]:
        """Add another particle to this inlet."""
        if not isinstance(particle, MockParticle):
            raise TypeError("AddParticle requires a particle definition.")
        self._properties.append(MockParticleInletProperties(particle))
        return self._properties

    def RemoveParticle(self, particle: MockParticle | str) -> None:
        """Remove a particle from this inlet."""
        name = particle if isinstance(particle, str) else particle.GetName()
        kept: list[MockParticleInletProperties] = []
        removed = False
        for properties in self._properties:
            current = properties.GetParticle()
            if current is particle or current.GetName() == name:
                removed = True
                continue
            kept.append(properties)
        if not removed:
            raise ValueError(f"Particle {name!r} is not on this inlet.")
        self._properties = kept

    def GetInputPropertiesList(self) -> list[MockParticleInletProperties]:
        """Return the inlet property list."""
        return self._properties

    def GetInjectionDuration(self, unit: str | None = None) -> float:
        """Return the injection duration."""
        return self._injection_duration

    def SetInjectionDuration(self, value: float, unit: str | None = None) -> None:
        """Set the injection duration in seconds."""
        self._injection_duration = float(value)

    def GetStartTime(self, unit: str | None = None) -> float:
        """Return the inlet start time in seconds."""
        if unit not in (None, "s"):
            raise ValueError(f"Unsupported time unit {unit!r}. Expected 's'.")
        return self._start_time

    def SetStartTime(self, value: float, unit: str | None = None) -> None:
        """Set the inlet start time in seconds."""
        if unit not in (None, "s"):
            raise ValueError(f"Unsupported time unit {unit!r}. Expected 's'.")
        self._start_time = float(value)

    def GetStopTime(self, unit: str | None = None) -> float:
        """Return the inlet stop time in seconds."""
        if unit not in (None, "s"):
            raise ValueError(f"Unsupported time unit {unit!r}. Expected 's'.")
        return self._stop_time

    def SetStopTime(self, value: float, unit: str | None = None) -> None:
        """Set the inlet stop time in seconds."""
        if unit not in (None, "s"):
            raise ValueError(f"Unsupported time unit {unit!r}. Expected 's'.")
        self._stop_time = float(value)

    def GetTargetNormalVelocity(self, unit: str | None = None) -> float:
        """Return the target injection speed in meters per second."""
        if unit not in (None, "m/s"):
            raise ValueError(f"Unsupported velocity unit {unit!r}. Expected 'm/s'.")
        return self._target_velocity

    def SetTargetNormalVelocity(self, value: float, unit: str | None = None) -> None:
        """Set the target injection speed in meters per second."""
        if unit not in (None, "m/s"):
            raise ValueError(f"Unsupported velocity unit {unit!r}. Expected 'm/s'.")
        self._target_velocity = float(value)

    def SetUseTargetNormalVelocity(self, value: bool) -> None:
        """Enable or disable the target injection speed."""
        self._use_target_velocity = bool(value)

    def GetUseTargetNormalVelocity(self) -> bool:
        """Return whether the target injection speed is used."""
        return self._use_target_velocity

    def GetPeriodic(self) -> bool:
        """Return whether the inlet repeats."""
        return self._periodic

    def SetPeriodic(self, value: bool) -> None:
        """Enable or disable a repeating inlet."""
        self._periodic = bool(value)

    def GetPeriod(self, unit: str | None = None) -> float:
        """Return the repeat period in seconds."""
        if unit not in (None, "s"):
            raise ValueError(f"Unsupported time unit {unit!r}. Expected 's'.")
        return self._period

    def SetPeriod(self, value: float, unit: str | None = None) -> None:
        """Set the repeat period in seconds."""
        if unit not in (None, "s"):
            raise ValueError(f"Unsupported time unit {unit!r}. Expected 's'.")
        self._period = float(value)

    def _velocity(self, unit: str | None) -> None:
        """Reject a velocity unit other than meters per second."""
        if unit not in (None, "m/s"):
            raise ValueError(f"Unsupported velocity unit {unit!r}. Expected 'm/s'.")

    def GetUxLocal(self, unit: str | None = None) -> float:
        """Return the local x velocity in meters per second."""
        self._velocity(unit)
        return self._ux

    def SetUxLocal(self, value: float, unit: str | None = None) -> None:
        """Set the local x velocity in meters per second."""
        self._velocity(unit)
        self._ux = float(value)

    def GetUzLocal(self, unit: str | None = None) -> float:
        """Return the local z velocity in meters per second."""
        self._velocity(unit)
        return self._uz

    def SetUzLocal(self, value: float, unit: str | None = None) -> None:
        """Set the local z velocity in meters per second."""
        self._velocity(unit)
        self._uz = float(value)

    def GetStopAllAtStopTime(self) -> bool:
        """Return whether every particle stops at the stop time."""
        return self._stop_all

    def SetStopAllAtStopTime(self, value: bool) -> None:
        """Set whether every particle stops at the stop time."""
        self._stop_all = bool(value)

    def GetForcePacking(self) -> bool:
        """Return whether the inlet force-packs particles."""
        return self._force_packing

    def SetForcePacking(self, value: bool) -> None:
        """Set whether the inlet force-packs particles."""
        self._force_packing = bool(value)

    def GetSphInjectionEnabled(self) -> bool:
        """Return whether SPH injection is enabled."""
        return self._sph_injection

    def SetSphInjectionEnabled(self, value: bool) -> None:
        """Enable or disable SPH injection."""
        self._sph_injection = bool(value)

    def GetSphTemperature(self, unit: str | None = None) -> float:
        """Return the SPH temperature in kelvin."""
        _require_kelvin(unit)
        return self._sph_temperature

    def SetSphTemperature(self, value: float, unit: str | None = None) -> None:
        """Set the SPH temperature in kelvin."""
        _require_kelvin(unit)
        self._sph_temperature = float(value)


class MockVolumetricInlet:
    """Stand-in for a Rocky volumetric inlet."""

    def __init__(
        self,
        particle: MockParticle | None,
        name: str | None,
        mass: float,
        seed_coordinates: list[float] | None,
        geometries: list[Any] | None,
        use_geometries_to_compute: bool,
        box_center: list[float] | None,
        box_dimensions: list[float] | None,
        use_box_center_as_seed_point: bool,
    ) -> None:
        """Store the fill definition."""
        self._particle = particle
        self._name = name or "Volumetric Inlet"
        self._mass = float(mass)
        self._seed = seed_coordinates
        self._geometries = list(geometries or [])
        self._use_geometries = use_geometries_to_compute
        self._box_center = box_center
        self._box_dimensions = box_dimensions
        self._use_box_center = use_box_center_as_seed_point

    def GetName(self) -> str:
        """Return the inlet name."""
        return self._name

    def SetName(self, value: str) -> None:
        """Set the inlet name."""
        self._name = value

    def GetMass(self) -> float:
        """Return the fill mass in kilograms."""
        return self._mass

    def UsesGeometries(self) -> bool:
        """Return whether the fill is bounded by geometries."""
        return self._use_geometries

    def GetBoxCenter(self) -> list[float] | None:
        """Return the limit-box center in meters."""
        return self._box_center

    def GetBoxDimensions(self) -> list[float] | None:
        """Return the limit-box dimensions in meters."""
        return self._box_dimensions


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
        self._material: Any = None
        self._temperature_k = 298.15
        self._convert_yz = False
        self._split_method = "BODY"
        self._normal_angle = 16.0
        self._max_edge: float | None = None
        self._sph_boundary = "free_slip"
        self._contact_angle_deg = 0.0
        self._wear_model = "none"
        self._wear_ratio = 0.0

    def GetName(self) -> str:
        """Return the wall name."""
        return self._name

    def GetClassName(self) -> str:
        """Return the Rocky class name of the wall."""
        return "RAWall"

    def GetFilename(self) -> str:
        """Return the source file of the wall."""
        return self._filename

    def GetConvertYZ(self) -> bool:
        """Return whether the import swapped Y and Z."""
        return self._convert_yz

    def GetGeometrySplitMethod(self) -> str:
        """Return the CAD split method used at import."""
        return self._split_method

    def GetNormalAngleTolerance(self) -> float:
        """Return the CAD mesh angle tolerance in degrees."""
        return self._normal_angle

    def GetMaxEdgeSize(self) -> float | None:
        """Return the CAD mesh maximum edge size in meters."""
        return self._max_edge

    def GetValidSphBoundaryTypeValues(self) -> list[str]:
        """Return the SPH boundary types this stand-in accepts."""
        return ["free_slip", "no_slip_laminar", "no_slip_turbulent"]

    def GetSphBoundaryType(self) -> str:
        """Return the SPH boundary type."""
        return self._sph_boundary

    def SetSphBoundaryType(self, value: str) -> None:
        """Set the SPH boundary type."""
        valid = self.GetValidSphBoundaryTypeValues()
        if value not in valid:
            raise ValueError(f"Unknown SPH boundary type {value!r}. Expected one of {valid}.")
        self._sph_boundary = value

    def GetSurfaceTensionContactAngle(self, unit: str | None = None) -> float:
        """Return the contact angle in degrees."""
        if unit not in (None, "dega"):
            raise ValueError(f"Unsupported angle unit {unit!r}. Expected 'dega'.")
        return self._contact_angle_deg

    def SetSurfaceTensionContactAngle(self, value: float, unit: str | None = None) -> None:
        """Set the contact angle. Rocky's degree token is ``dega``."""
        if unit not in (None, "dega"):
            raise ValueError(f"Unsupported angle unit {unit!r}. Expected 'dega'.")
        self._contact_angle_deg = float(value)

    def GetWearModel(self) -> str:
        """Return the wear model."""
        return self._wear_model

    def SetWearModel(self, value: str) -> None:
        """Set the wear model."""
        if value not in ("none", "archard", "custom"):
            raise ValueError(f"Unknown wear model {value!r}.")
        self._wear_model = value

    def GetVolumeShearWorkRatio(self, unit: str | None = None) -> float:
        """Return the Archard volume per energy. Rocky's token is ``m3/J``."""
        if unit not in (None, "m3/J"):
            raise ValueError(f"Unsupported wear unit {unit!r}. Expected 'm3/J'.")
        return self._wear_ratio

    def SetVolumeShearWorkRatio(self, value: float, unit: str | None = None) -> None:
        """Set the Archard volume per energy. Rocky's token is ``m3/J``."""
        if unit not in (None, "m3/J"):
            raise ValueError(f"Unsupported wear unit {unit!r}. Expected 'm3/J'.")
        self._wear_ratio = float(value)

    def SetMaterial(self, value: Any) -> None:
        """Assign a solid material."""
        self._material = value

    def GetTemperature(self, unit: str | None = None) -> float:
        """Return the wall temperature in kelvin."""
        _require_kelvin(unit)
        return self._temperature_k

    def SetTemperature(self, value: float, unit: str | None = None) -> None:
        """Set the wall temperature in kelvin."""
        _require_kelvin(unit)
        self._temperature_k = float(value)

    def GetMaterial(self) -> Any:
        """Return the assigned material."""
        return self._material


class MockDomainSettings:
    """Stand-in for Rocky domain settings."""

    def __init__(self) -> None:
        """Initialize the domain."""
        self._use_boundary_limits = True
        self._min = (-DEFAULT_DOMAIN_LENGTH / 2.0, -1.0, -1.0)
        self._max = (DEFAULT_DOMAIN_LENGTH / 2.0, 1.0, 1.0)
        self._domain_type = "NONE"
        self._cartesian_periodic = "NONE"
        self._cylindrical_periodic = "Y"
        self._divisions = 1
        self._periodic_min = self._min
        self._periodic_max = self._max
        self._periodic_at_geometry = False

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

    def GetCartesianPeriodicDirections(self) -> str:
        """Return the Cartesian periodic directions."""
        return self._cartesian_periodic

    def SetCartesianPeriodicDirections(self, value: str) -> None:
        """Set the Cartesian periodic directions."""
        self._cartesian_periodic = value

    def GetCylindricalPeriodicDirections(self) -> str:
        """Return the cylindrical periodic axis."""
        return self._cylindrical_periodic

    def SetCylindricalPeriodicDirections(self, value: str) -> None:
        """Set the cylindrical periodic axis."""
        self._cylindrical_periodic = value

    def GetPeriodicLimitsMinCoordinates(self, unit: str | None = None) -> list[float]:
        """Return the minimum periodic coordinates."""
        factor = 1.0 / _LENGTH_TO_M.get(unit or "m", 1.0)
        return [value * factor for value in self._periodic_min]

    def SetPeriodicLimitsMinCoordinates(
        self, values: Any, unit: str | None = None, **kwargs: Any
    ) -> None:
        """Set the minimum periodic coordinates."""
        factor = _LENGTH_TO_M.get(unit or "m", 1.0)
        self._periodic_min = tuple(value * factor for value in _as_sequence(values))

    def GetPeriodicLimitsMaxCoordinates(self, unit: str | None = None) -> list[float]:
        """Return the maximum periodic coordinates."""
        factor = 1.0 / _LENGTH_TO_M.get(unit or "m", 1.0)
        return [value * factor for value in self._periodic_max]

    def SetPeriodicLimitsMaxCoordinates(
        self, values: Any, unit: str | None = None, **kwargs: Any
    ) -> None:
        """Set the maximum periodic coordinates."""
        factor = _LENGTH_TO_M.get(unit or "m", 1.0)
        self._periodic_max = tuple(value * factor for value in _as_sequence(values))

    def GetPeriodicAtGeometryLimits(self) -> bool:
        """Return whether periodic limits follow the geometry."""
        return self._periodic_at_geometry

    def SetPeriodicAtGeometryLimits(self, value: bool) -> None:
        """Set whether periodic limits follow the geometry."""
        self._periodic_at_geometry = bool(value)

    def GetNumberOfDivisions(self) -> int:
        """Return the number of cylindrical divisions."""
        return self._divisions

    def SetNumberOfDivisions(self, value: int | str) -> None:
        """Set the number of cylindrical divisions."""
        count = int(value)
        if count < 1:
            raise ValueError("Number of divisions must be at least 1.")
        self._divisions = count


class MockExecutionSettings:
    """Stand-in for local or MPI processor settings."""

    def __init__(self, processors: int = 1) -> None:
        """Initialize the processor count."""
        self._processors = processors
        self._processing_unit = "CPU"
        self._target_gpus: list[int] = []
        self._gpu_slicing = "X"

    def GetNumberOfProcessors(self) -> int:
        """Return the processor count for this execution path."""
        return self._processors

    def SetNumberOfProcessors(self, value: int | str) -> None:
        """Set the processor count for this execution path."""
        self._processors = int(value)

    def GetValidProcessingUnitValues(self) -> list[str]:
        """Return the processing units this settings object accepts."""
        return ["CPU", "GPU"]

    def GetProcessingUnit(self) -> str:
        """Return the processing unit."""
        return self._processing_unit

    def SetProcessingUnit(self, value: str) -> None:
        """Set the processing unit."""
        if value not in self.GetValidProcessingUnitValues():
            raise ValueError(f"Unknown processing unit {value!r}.")
        self._processing_unit = value

    def GetTargetGpus(self) -> list[int]:
        """Return the selected GPU ids."""
        return list(self._target_gpus)

    def SetTargetGpus(self, gpus: list[int]) -> None:
        """Set the GPU ids."""
        self._target_gpus = [int(gpu) for gpu in gpus]

    def GetValidMultiGpuSlicingDirectionValues(self) -> list[str]:
        """Return the slicing directions this settings object accepts."""
        return ["X", "Y", "Z"]

    def GetMultiGpuSlicingDirection(self) -> str:
        """Return the slicing direction."""
        return self._gpu_slicing

    def SetMultiGpuSlicingDirection(self, value: str) -> None:
        """Set the slicing direction."""
        if value not in self.GetValidMultiGpuSlicingDirectionValues():
            raise ValueError(f"Unknown slicing direction {value!r}.")
        self._gpu_slicing = value


class MockSimulatorRun:
    """Stand-in for the Rocky solver / simulator run object."""

    def __init__(self) -> None:
        """Initialize the solver."""
        self._duration = DEFAULT_DURATION
        self._time_interval = DEFAULT_DURATION / 20.0
        self._processors = 1
        self._local_settings = MockExecutionSettings(1)
        self._mpi_settings = MockExecutionSettings(1)
        self._curves_frequency = 1
        self._execution_mode = "local"
        self._gpu = 0
        self._fixed_timestep = 1e-4
        self._timestep_model = "variable"
        self._use_fixed_timestep = False
        self._release_without_overlap = False
        self._breakage_delay = 0.0
        self._standard_outputs: dict[tuple[str, ...], bool] = {}
        self._module_outputs: dict[tuple[str, ...], bool] = {}
        self._processing_unit = "CPU"
        self._target_gpus: list[int] = []
        self._gpu_slicing = "X"

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

    def GetLocalExecutionSettings(self) -> MockExecutionSettings:
        """Return the local execution settings."""
        return self._local_settings

    def GetMpiSettings(self) -> MockExecutionSettings:
        """Return the MPI execution settings."""
        return self._mpi_settings

    def GetSolverCurvesFrequency(self) -> int:
        """Return how often solver curves are stored."""
        return self._curves_frequency

    def SetSolverCurvesFrequency(self, value: int | str) -> None:
        """Set how often solver curves are stored."""
        frequency = int(value)
        if frequency < 1:
            raise ValueError("Solver curves frequency must be at least 1.")
        self._curves_frequency = frequency

    def GetExecutionMode(self) -> str:
        """Return the execution mode."""
        return self._execution_mode

    def SetExecutionMode(self, value: str) -> None:
        """Set the execution mode."""
        if value not in self.GetValidExecutionModeValues():
            legal = ", ".join(self.GetValidExecutionModeValues())
            raise ValueError(f"Invalid execution mode {value!r}. Expected one of {legal}.")
        self._execution_mode = value

    def GetValidExecutionModeValues(self) -> list[str]:
        """Return the supported execution modes."""
        return ["local", "single_mpi", "multi_mpi"]

    def GetTargetGpu(self) -> int:
        """Return the target GPU index."""
        return self._gpu

    def SetTargetGpu(self, value: int) -> None:
        """Set the target GPU index."""
        self._gpu = int(value)

    def GetFixedTimestep(self, unit: str | None = None) -> float:
        """Return the fixed timestep in seconds."""
        if unit not in (None, "s"):
            raise ValueError(f"Unsupported time unit {unit!r}. Expected 's'.")
        return self._fixed_timestep

    def SetFixedTimestep(self, value: float, unit: str | None = None) -> None:
        """Set the fixed timestep in seconds."""
        if unit not in (None, "s"):
            raise ValueError(f"Unsupported time unit {unit!r}. Expected 's'.")
        self._fixed_timestep = float(value)

    def GetTimestepModel(self) -> str:
        """Return the timestep model."""
        return self._timestep_model

    def SetTimestepModel(self, value: str) -> None:
        """Set the timestep model."""
        if value not in ("constant", "variable"):
            raise ValueError(f"Unknown timestep model {value!r}.")
        self._timestep_model = value

    def GetUseFixedTimestep(self) -> bool:
        """Return whether the fixed timestep is used."""
        return self._use_fixed_timestep

    def SetUseFixedTimestep(self, value: bool) -> None:
        """Enable or disable the fixed timestep."""
        self._use_fixed_timestep = bool(value)

    def GetReleaseParticlesWithoutOverlapCheck(self) -> bool:
        """Return whether particles are released without an overlap check."""
        return self._release_without_overlap

    def SetReleaseParticlesWithoutOverlapCheck(self, value: bool) -> None:
        """Set whether particles are released without an overlap check."""
        self._release_without_overlap = bool(value)

    def GetBreakageDelayAfterRelease(self, unit: str | None = None) -> float:
        """Return the breakage delay after release in seconds."""
        if unit not in (None, "s"):
            raise ValueError(f"Unsupported time unit {unit!r}. Expected 's'.")
        return self._breakage_delay

    def SetBreakageDelayAfterRelease(self, value: float, unit: str | None = None) -> None:
        """Set the breakage delay after release in seconds."""
        if unit not in (None, "s"):
            raise ValueError(f"Unsupported time unit {unit!r}. Expected 's'.")
        self._breakage_delay = float(value)

    @staticmethod
    def GetAvailableStandardOutputProperties() -> dict[str, list[str]]:
        """Return a small set of standard output names."""
        return {"Particles": ["Count", "Mass Flow In", "Mass Flow Out"]}

    @staticmethod
    def GetAvailableModulesOutputProperties() -> dict[str, list[str]]:
        """Return a small set of module output names."""
        return {"Breakage": ["Fragments"]}

    def SetStandardOutputPropertyEnabled(self, *output_property: str, enabled: bool = True) -> None:
        """Enable or disable a standard output property."""
        if not output_property:
            raise ValueError("An output property name is required.")
        self._standard_outputs[tuple(output_property)] = bool(enabled)

    def GetStandardOutputPropertyEnabled(self, *output_property: str) -> bool:
        """Return whether a standard output property is enabled."""
        return self._standard_outputs.get(tuple(output_property), False)

    def SetModulesOutputPropertyEnabled(self, *output_property: str, enabled: bool = True) -> None:
        """Enable or disable a module output property."""
        if not output_property:
            raise ValueError("An output property name is required.")
        self._module_outputs[tuple(output_property)] = bool(enabled)

    def GetModulesOutputPropertyEnabled(self, *output_property: str) -> bool:
        """Return whether a module output property is enabled."""
        return self._module_outputs.get(tuple(output_property), False)

    def GetValidProcessingUnitValues(self) -> list[str]:
        """Return the processing units this stand-in accepts."""
        return ["CPU", "GPU"]

    def GetProcessingUnit(self) -> str:
        """Return the processing unit."""
        return self._processing_unit

    def SetProcessingUnit(self, value: str) -> None:
        """Set the processing unit."""
        valid = self.GetValidProcessingUnitValues()
        if value not in valid:
            raise ValueError(f"Unknown processing unit {value!r}. Expected one of {valid}.")
        self._processing_unit = value

    def GetTargetGpus(self) -> list[int]:
        """Return the selected GPU ids."""
        return list(self._target_gpus)

    def SetTargetGpus(self, gpus: list[int]) -> None:
        """Set the GPU ids used by a local simulation."""
        self._target_gpus = [int(gpu) for gpu in gpus]

    def GetValidMultiGpuSlicingDirectionValues(self) -> list[str]:
        """Return the multi-GPU slicing directions this stand-in accepts."""
        return ["X", "Y", "Z"]

    def GetMultiGpuSlicingDirection(self) -> str:
        """Return the multi-GPU slicing direction."""
        return self._gpu_slicing

    def SetMultiGpuSlicingDirection(self, value: str) -> None:
        """Set the multi-GPU slicing direction."""
        valid = self.GetValidMultiGpuSlicingDirectionValues()
        if value not in valid:
            raise ValueError(f"Unknown slicing direction {value!r}. Expected one of {valid}.")
        self._gpu_slicing = value


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
        if (
            isinstance(angular_velocity, tuple)
            and len(angular_velocity) == 2
            and isinstance(angular_velocity[1], str)
            and angular_velocity[1] != "rad/s"
        ):
            raise ValueError(
                f"Unsupported angular velocity unit {angular_velocity[1]!r}. Expected 'rad/s'."
            )
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

    def GetCurveNames(self) -> list[str]:
        """Return the curve names this selection can read."""
        return self._study.GetParticles().GetCurveNames()

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
        magnitude: Any = None,
        **kwargs: Any,
    ) -> MockCubeProcess:
        """Create a cube selection process."""
        process = MockCubeProcess(name or f"Cube Selection {len(self._study._selections) + 1}")
        process._study = self._study
        if center is not None:
            process.SetCenter(center)
        if magnitude is not None:
            sx, sy, sz = _as_sequence(magnitude)
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


class MockPhysics:
    """Stand-in for Rocky physics, including the gravity vector."""

    def __init__(self) -> None:
        """Initialize gravity at rest."""
        self._gravity = [0.0, 0.0, 0.0]
        self._normal_force = "damped_hertzian"
        self._tangential_force = "mindlin_deresiewicz"
        self._rolling_resistance = "none"
        self._adhesion = "none"
        self._restitution_model = "constant"
        self._thermal = False

    def GetGravityXDirection(self, unit: str | None = None) -> float:
        """Return the X component of gravity."""
        return self._gravity[0]

    def SetGravityXDirection(self, value: float, unit: str | None = None) -> None:
        """Set the X component of gravity."""
        self._gravity[0] = float(value)

    def GetGravityYDirection(self, unit: str | None = None) -> float:
        """Return the Y component of gravity."""
        return self._gravity[1]

    def SetGravityYDirection(self, value: float, unit: str | None = None) -> None:
        """Set the Y component of gravity."""
        self._gravity[1] = float(value)

    def GetGravityZDirection(self, unit: str | None = None) -> float:
        """Return the Z component of gravity."""
        return self._gravity[2]

    def SetGravityZDirection(self, value: float, unit: str | None = None) -> None:
        """Set the Z component of gravity."""
        self._gravity[2] = float(value)

    def GetNormalForceModel(self) -> str:
        """Return the normal force model."""
        return self._normal_force

    def SetNormalForceModel(self, value: str) -> None:
        """Set the normal force model."""
        self._normal_force = value

    def GetTangentialForceModel(self) -> str:
        """Return the tangential force model."""
        return self._tangential_force

    def SetTangentialForceModel(self, value: str) -> None:
        """Set the tangential force model."""
        self._tangential_force = value

    def GetRollingResistanceModel(self) -> str:
        """Return the rolling resistance model."""
        return self._rolling_resistance

    def SetRollingResistanceModel(self, value: str) -> None:
        """Set the rolling resistance model."""
        self._rolling_resistance = value

    def GetAdhesionModel(self) -> str:
        """Return the adhesion model."""
        return self._adhesion

    def SetAdhesionModel(self, value: str) -> None:
        """Set the adhesion model."""
        self._adhesion = value

    def GetRestitutionModel(self) -> str:
        """Return the restitution model."""
        return self._restitution_model

    def SetRestitutionModel(self, value: str) -> None:
        """Set the restitution model."""
        self._restitution_model = value

    def GetEnableThermalModel(self) -> bool:
        """Return whether the thermal model is enabled."""
        return self._thermal

    def SetEnableThermalModel(self, value: bool) -> None:
        """Enable or disable the thermal model."""
        self._thermal = bool(value)


class MockExportToolkit:
    """Stand-in for the Rocky export toolkit."""

    def __init__(self, study: MockStudy) -> None:
        """Initialize the toolkit for one study."""
        self._study = study

    def ExportToSTL(
        self,
        stl_filename: str,
        entities: Any,
        time_to_export: Any = None,
        output_unit: str | None = None,
    ) -> None:
        """Write a stand-in STL that records which entities were requested."""
        items = entities if isinstance(entities, list) else [entities]
        names = []
        for item in items:
            get_name = getattr(item, "GetName", None)
            names.append(get_name() if get_name is not None else str(item))
        path = Path(stl_filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("solid mock\n" + "\n".join(names) + "\nendsolid mock\n", encoding="utf-8")


class MockFeedConveyor:
    """Stand-in for a Rocky feed conveyor."""

    def __init__(self, name: str) -> None:
        """Initialize a stopped conveyor."""
        self._name = name
        self._speed = 0.0
        self._start_time = 0.0
        self._stop_time = 0.0
        self._acceleration = 0.0
        self._deceleration = 0.0
        self._profile = "Three Rolls"
        self._width = 1.0
        self._loading_length = 0.0
        self._drop = [0.0, 0.0, 0.0]
        self._skirtboard = 0.0
        self._transition = 0.0
        self._triangle = 0.05
        self._thickness = 0.01
        self._diameter = 0.3
        self._wall_thickness = 0.01
        self._face_width = 0.3
        self._alignment_rad = 0.0
        self._return_belt_rad = 0.0
        self._out_of_plane = 0.0
        self._temperature_k = 298.15
        self._incline_rad = 0.0
        self._horizontal_offset = 0.0
        self._vertical_offset = 0.0
        self._material: Any = None

    def GetName(self) -> str:
        """Return the conveyor name."""
        return self._name

    def SetName(self, value: str) -> None:
        """Set the conveyor name."""
        self._name = value

    def SetBeltSpeed(self, value: float, unit: str | None = None) -> None:
        """Set the belt speed in meters per second."""
        self._speed = float(value)

    def GetBeltSpeed(self, unit: str | None = None) -> float:
        """Return the belt speed in meters per second."""
        return self._speed

    def GetTemperature(self, unit: str | None = None) -> float:
        """Return the belt temperature in kelvin."""
        _require_kelvin(unit)
        return self._temperature_k

    def SetTemperature(self, value: float, unit: str | None = None) -> None:
        """Set the belt temperature in kelvin."""
        _require_kelvin(unit)
        self._temperature_k = float(value)

    def SetBeginningStartTime(self, value: float, unit: str | None = None) -> None:
        """Set when the belt starts, in seconds."""
        self._start_time = float(value)

    def GetBeginningStartTime(self, unit: str | None = None) -> float:
        """Return when the belt starts, in seconds."""
        return self._start_time

    def SetAccelerationPeriod(self, value: float, unit: str | None = None) -> None:
        """Set how long the belt takes to reach speed, in seconds."""
        self._acceleration = float(value)

    def GetAccelerationPeriod(self, unit: str | None = None) -> float:
        """Return how long the belt takes to reach speed, in seconds."""
        return self._acceleration

    def SetBeginningStopTime(self, value: float, unit: str | None = None) -> None:
        """Set when the belt starts to stop, in seconds."""
        self._stop_time = float(value)

    def GetBeginningStopTime(self, unit: str | None = None) -> float:
        """Return when the belt starts to stop, in seconds."""
        return self._stop_time

    def SetDecelerationPeriod(self, value: float, unit: str | None = None) -> None:
        """Set how long the belt takes to stop, in seconds."""
        self._deceleration = float(value)

    def GetDecelerationPeriod(self, unit: str | None = None) -> float:
        """Return how long the belt takes to stop, in seconds."""
        return self._deceleration

    def GetValidBeltProfileNames(self) -> list[str]:
        """Return the belt profiles this stand-in accepts."""
        return ["One Roll", "Two Rolls", "Three Rolls", "Five Rolls"]

    def SetBeltProfile(self, belt_profile_name: str) -> str:
        """Set the belt profile by name."""
        if belt_profile_name not in self.GetValidBeltProfileNames():
            raise ValueError(
                f"Unknown belt profile {belt_profile_name!r}. "
                f"Expected one of {self.GetValidBeltProfileNames()}."
            )
        self._profile = belt_profile_name
        return belt_profile_name

    def GetBeltProfileName(self) -> str:
        """Return the belt profile name."""
        return self._profile

    def SetBeltWidth(self, value: float, unit: str | None = None) -> None:
        """Set the belt width in meters."""
        self._width = float(value)

    def GetBeltWidth(self, unit: str | None = None) -> float:
        """Return the belt width in meters."""
        return self._width

    def SetLoadingLength(self, value: float, unit: str | None = None) -> None:
        """Set the loading length in meters."""
        self._loading_length = float(value)

    def GetLoadingLength(self, unit: str | None = None) -> float:
        """Return the loading length in meters."""
        return self._loading_length

    def SetDropBoxLength(self, value: float, unit: str | None = None) -> None:
        """Set the drop box length in meters."""
        self._drop[0] = float(value)

    def GetDropBoxLength(self, unit: str | None = None) -> float:
        """Return the drop box length in meters."""
        return self._drop[0]

    def SetDropBoxWidth(self, value: float, unit: str | None = None) -> None:
        """Set the drop box width in meters."""
        self._drop[1] = float(value)

    def GetDropBoxWidth(self, unit: str | None = None) -> float:
        """Return the drop box width in meters."""
        return self._drop[1]

    def SetDropBoxHeight(self, value: float, unit: str | None = None) -> None:
        """Set the drop box height in meters."""
        self._drop[2] = float(value)

    def GetDropBoxHeight(self, unit: str | None = None) -> float:
        """Return the drop box height in meters."""
        return self._drop[2]

    def SetSkirtboardHeight(self, value: float, unit: str | None = None) -> None:
        """Set the skirtboard height in meters."""
        self._skirtboard = float(value)

    def GetSkirtboardHeight(self, unit: str | None = None) -> float:
        """Return the skirtboard height in meters."""
        return self._skirtboard

    def SetTransitionLength(self, value: float, unit: str | None = None) -> None:
        """Set the transition length in meters."""
        self._transition = float(value)

    def GetTransitionLength(self, unit: str | None = None) -> float:
        """Return the transition length in meters."""
        return self._transition

    def SetTriangleSize(self, value: float, unit: str | None = None) -> None:
        """Set the belt mesh triangle size in meters."""
        self._triangle = float(value)

    def GetTriangleSize(self, unit: str | None = None) -> float:
        """Return the belt mesh triangle size in meters."""
        return self._triangle

    def SetBeltThickness(self, value: float, unit: str | None = None) -> None:
        """Set the belt thickness in meters."""
        self._thickness = float(value)

    def GetBeltThickness(self, unit: str | None = None) -> float:
        """Return the belt thickness in meters."""
        return self._thickness

    def SetDiameter(self, value: float, unit: str | None = None) -> None:
        """Set the pulley diameter in meters."""
        self._diameter = float(value)

    def GetDiameter(self, unit: str | None = None) -> float:
        """Return the pulley diameter in meters."""
        return self._diameter

    def SetWallThickness(self, value: float, unit: str | None = None) -> None:
        """Set the conveyor wall thickness in meters."""
        self._wall_thickness = float(value)

    def GetWallThickness(self, unit: str | None = None) -> float:
        """Return the conveyor wall thickness in meters."""
        return self._wall_thickness

    def SetFaceWidth(self, value: float, unit: str | None = None) -> None:
        """Set the pulley face width in meters."""
        self._face_width = float(value)

    def GetFaceWidth(self, unit: str | None = None) -> float:
        """Return the pulley face width in meters."""
        return self._face_width

    def SetAlignmentAngle(self, value: float, unit: str | None = None) -> None:
        """Set the conveyor alignment angle. Degrees use Rocky's ``dega`` token."""
        angle = float(value)
        if unit in ("deg", "dega"):
            angle = math.radians(angle)
        elif unit not in (None, "rad"):
            raise ValueError(f"Unsupported angle unit {unit!r}.")
        self._alignment_rad = angle

    def SetReturnBeltAngle(self, value: float, unit: str | None = None) -> None:
        """Set the return belt angle. Rocky's degree token is ``dega``."""
        angle = float(value)
        if unit in ("deg", "dega"):
            angle = math.radians(angle)
        elif unit not in (None, "rad"):
            raise ValueError(f"Unsupported angle unit {unit!r}.")
        self._return_belt_rad = angle

    def SetOutOfPlaneOffset(self, value: float, unit: str | None = None) -> None:
        """Set the out-of-plane offset in meters."""
        if unit not in (None, "m"):
            raise ValueError(f"Unsupported length unit {unit!r}. Expected 'm'.")
        self._out_of_plane = float(value)

    def GetOutOfPlaneOffset(self, unit: str | None = None) -> float:
        """Return the out-of-plane offset in meters."""
        if unit not in (None, "m"):
            raise ValueError(f"Unsupported length unit {unit!r}. Expected 'm'.")
        return self._out_of_plane

    def GetReturnBeltAngle(self, unit: str | None = None) -> float:
        """Return the return belt angle in radians, or degrees when asked."""
        if unit in ("deg", "dega"):
            return math.degrees(self._return_belt_rad)
        return self._return_belt_rad

    def GetAlignmentAngle(self, unit: str | None = None) -> float:
        """Return the alignment angle in radians, or degrees when asked."""
        if unit in ("deg", "dega"):
            return math.degrees(self._alignment_rad)
        return self._alignment_rad

    def SetBeltInclineAngle(self, value: float, unit: str | None = None) -> None:
        """Set the belt incline. Degrees are accepted as ``deg``."""
        angle = float(value)
        if unit in ("deg", "dega"):
            angle = math.radians(angle)
        elif unit not in (None, "rad"):
            raise ValueError(f"Unsupported angle unit {unit!r}.")
        self._incline_rad = angle

    def GetBeltInclineAngle(self, unit: str | None = None) -> float:
        """Return the belt incline in radians, or degrees when asked."""
        if unit in ("deg", "dega"):
            return math.degrees(self._incline_rad)
        return self._incline_rad

    def SetHorizontalOffset(self, value: float, unit: str | None = None) -> None:
        """Set the horizontal offset in meters."""
        self._horizontal_offset = float(value)

    def GetHorizontalOffset(self, unit: str | None = None) -> float:
        """Return the horizontal offset in meters."""
        return self._horizontal_offset

    def SetVerticalOffset(self, value: float, unit: str | None = None) -> None:
        """Set the vertical offset in meters."""
        self._vertical_offset = float(value)

    def GetVerticalOffset(self, unit: str | None = None) -> float:
        """Return the vertical offset in meters."""
        return self._vertical_offset

    def SetMaterial(self, value: Any) -> None:
        """Assign a solid material to the belt."""
        self._material = value

    def GetMaterial(self) -> Any:
        """Return the belt material."""
        return self._material


class MockStudy:
    """Stand-in for a Rocky study, the container of a DEM simulation."""

    def __init__(self, name: str = "Study 1") -> None:
        """Initialize an empty study."""
        self._name = name
        self._particles: list[MockParticle] = []
        self._surfaces: list[MockSurface] = []
        self._inlets: list[MockParticleInlet | MockVolumetricInlet] = []
        self._outlets: list[MockOutlet] = []
        self._walls: list[MockWall] = []
        self._conveyors: list[MockFeedConveyor] = []
        self._selections: list[MockSelectionProcess] = []
        self._domain = MockDomainSettings()
        self._physics = MockPhysics()
        self._materials = MockMaterialCollection()
        self._interactions = MockMaterialsInteractionCollection()
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

    def CreateFeedConveyor(self) -> MockFeedConveyor:
        """Create a feed conveyor."""
        return self._new_conveyor("Feed Conveyor")

    def CreateReceivingConveyor(self) -> MockFeedConveyor:
        """Create a receiving conveyor."""
        return self._new_conveyor("Receiving Conveyor")

    def _new_conveyor(self, label: str) -> MockFeedConveyor:
        """Create a conveyor with the next index."""
        conveyor = MockFeedConveyor(f"{label} {len(self._conveyors) + 1}")
        conveyor._is_feed = label.startswith("Feed")
        self._conveyors.append(conveyor)
        return conveyor

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
        """Create a particle inlet on a surface or a feed conveyor."""
        if entry_point is None:
            entry_point = kwargs.get("entry_point")
        if particle is None:
            particle = kwargs.get("particle")
        if entry_point is None or particle is None:
            raise ValueError("CreateParticleInlet requires an entry_point and a particle.")
        if isinstance(entry_point, MockFeedConveyor):
            if not getattr(entry_point, "_is_feed", False):
                raise TypeError("A particle inlet entry point must be a feed conveyor.")
        elif not isinstance(entry_point, MockSurface):
            raise TypeError(
                "The entry point of a particle inlet must be a surface or a feed conveyor."
            )
        if not isinstance(particle, MockParticle):
            raise TypeError("The particle of a particle inlet must be a particle definition.")
        inlet = MockParticleInlet(entry_point, particle)
        self._inlets.append(inlet)
        return inlet

    def CreateVolumetricInlet(
        self,
        particle: MockParticle | None = None,
        name: str | None = None,
        mass: float = 100.0,
        seed_coordinates: list[float] | None = None,
        geometries: list[Any] | None = None,
        use_geometries_to_compute: bool = False,
        box_center: list[float] | None = None,
        box_dimensions: list[float] | None = None,
        use_box_center_as_seed_point: bool = False,
    ) -> MockVolumetricInlet:
        """Create a volumetric fill inlet."""
        if particle is not None and not isinstance(particle, MockParticle):
            raise TypeError("The particle of a volumetric inlet must be a particle definition.")
        inlet = MockVolumetricInlet(
            particle,
            name,
            mass,
            seed_coordinates,
            geometries,
            use_geometries_to_compute,
            box_center,
            box_dimensions,
            use_box_center_as_seed_point,
        )
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
        wall._convert_yz = bool(convert_yz)
        wall._split_method = str(kwargs.get("geometry_split_method", "BODY"))
        if "normal_angle_tolerance" in kwargs:
            wall._normal_angle = float(kwargs["normal_angle_tolerance"])
        if kwargs.get("max_edge_size") is not None:
            wall._max_edge = float(kwargs["max_edge_size"])
        self._walls.append(wall)
        return [wall]

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def GetDomainSettings(self) -> MockDomainSettings:
        """Return the domain settings."""
        return self._domain

    def GetPhysics(self) -> MockPhysics:
        """Return the physics settings."""
        return self._physics

    def GetMaterialCollection(self) -> MockMaterialCollection:
        """Return the material collection."""
        return self._materials

    def GetMaterialsInteractionCollection(self) -> MockMaterialsInteractionCollection:
        """Return the material interaction collection."""
        return self._interactions

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

    def GetExportToolkit(self) -> MockExportToolkit:
        """Return the export toolkit."""
        return MockExportToolkit(self)

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
        self._simulating = False

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

    def ExtendSimulation(self, extension_amount: float = 0, **kwargs: Any) -> None:
        """Extend the solver duration, and the finished time set when results exist."""
        extra = float(extension_amount)
        if extra < 0:
            raise ValueError("extension_amount must be zero or positive.")
        duration = self._solver.GetSimulationDuration() + extra
        self._solver.SetSimulationDuration(duration)
        if self._has_results and not self._simulating:
            self._elapsed = duration
            self._progress = 100.0

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
            getter = getattr(inlet, "GetInputPropertiesList", None)
            if getter is None:
                continue
            for properties in getter():
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
