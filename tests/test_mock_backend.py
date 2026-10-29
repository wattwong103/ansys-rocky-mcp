"""Tests for the in-memory Rocky simulator."""

from __future__ import annotations

import math

import pytest

from ansys_rocky_mcp.backends.mock_backend import MockBackend, MockRockyApplication


def test_application_lifecycle() -> None:
    """The mock application creates, exposes and closes projects."""
    app = MockRockyApplication()
    assert app.GetVersion().endswith("(mock)")
    assert app.GetProject() is None

    project = app.CreateProject()
    assert app.GetProject() is project

    app.CloseProject()
    assert app.GetProject() is None


def test_study_defaults() -> None:
    """A fresh study exposes the expected default settings."""
    app = MockRockyApplication()
    study = app.CreateProject().GetStudy()

    assert study.GetSolver().GetSimulationDuration("s") == pytest.approx(2.0)
    assert study.GetDomainSettings().GetUseBoundaryLimits() is True
    assert study.GetParticles() is not None
    assert study.GetUserProcessCollection() is not None


def test_particle_inlet_requires_valid_arguments() -> None:
    """The inlet rejects a missing entry point or particle."""
    app = MockRockyApplication()
    study = app.CreateProject().GetStudy()

    with pytest.raises(ValueError):
        study.CreateParticleInlet(None, None)

    surface = study.CreateCircularSurface()
    with pytest.raises(TypeError):
        study.CreateParticleInlet(surface, surface)


def test_inlet_mass_flow_roundtrip() -> None:
    """The mass flow rate is stored and read back in the requested unit."""
    app = MockRockyApplication()
    study = app.CreateProject().GetStudy()
    surface = study.CreateCircularSurface()
    particle = study.CreateParticle()

    inlet = study.CreateParticleInlet(entry_point=surface, particle=particle)
    properties = inlet.GetInputPropertiesList()[0]
    properties.SetMassFlowRate(1000.0, unit="t/h")

    assert properties.GetMassFlowRate("t/h") == pytest.approx(1000.0)
    assert properties.GetMassFlowRate("kg/s") == pytest.approx(1000.0 / 3.6)


def test_simulation_produces_results() -> None:
    """A simulation fills the particle count curve."""
    app = MockRockyApplication()
    study = app.CreateProject().GetStudy()
    surface = study.CreateCircularSurface()
    surface.SetMaxRadius(0.3, unit="m")
    particle = study.CreateParticle()
    particle.SetMinimumSize(0.05, unit="m")

    inlet = study.CreateParticleInlet(entry_point=surface, particle=particle)
    inlet.GetInputPropertiesList()[0].SetMassFlowRate(1000.0, unit="t/h")

    study.GetSolver().SetSimulationDuration(2.0, unit="s")
    study.StartSimulation(non_blocking=False)

    assert study.HasResults() is True
    assert study.IsSimulating() is False
    assert study.GetProgress() == pytest.approx(100.0)

    times, values = study.GetParticles().GetNumpyCurve("Particles Count")
    assert len(times) == len(values)
    assert len(times) > 1
    assert values[-1] > values[0]


def test_curves_require_results() -> None:
    """Requesting a curve before running raises a clear error."""
    app = MockRockyApplication()
    study = app.CreateProject().GetStudy()
    with pytest.raises(RuntimeError, match="rocky_run_simulation"):
        study.GetParticles().GetNumpyCurve("Particles Count")


def test_mass_flow_out_starts_after_transit() -> None:
    """The outlet flow only starts once particles reach the outlet."""
    app = MockRockyApplication()
    study = app.CreateProject().GetStudy()
    surface = study.CreateCircularSurface()
    surface.SetMaxRadius(0.3, unit="m")
    particle = study.CreateParticle()
    particle.SetMinimumSize(0.05, unit="m")
    inlet = study.CreateParticleInlet(entry_point=surface, particle=particle)
    inlet.GetInputPropertiesList()[0].SetMassFlowRate(1000.0, unit="t/h")

    study.GetDomainSettings().SetCoordinateLimitsMinValues([-5.0, -1.5, -1.5], unit="m")
    study.GetDomainSettings().SetCoordinateLimitsMaxValues([5.0, 1.5, 1.5], unit="m")
    study.GetSolver().SetSimulationDuration(2.0, unit="s")
    study.StartSimulation(non_blocking=False)

    derived = study._derived()
    assert derived["transit"] > 2.0

    times, out_flow = study.GetParticles().GetNumpyCurve("Particles Mass Flow Out")
    assert all(value == pytest.approx(0.0) for value in out_flow)

    _, in_flow = study.GetParticles().GetNumpyCurve("Particles Mass Flow In")
    assert in_flow[-1] == pytest.approx(1000.0)

    _, in_flow_kg_s = study.GetParticles().GetNumpyCurve("Particles Mass Flow In", unit="kg/s")
    assert in_flow_kg_s[-1] == pytest.approx(1000.0 / 3.6)


def test_velocity_scales_with_mass_flow() -> None:
    """Doubling the mass flow doubles the inlet velocity."""
    app = MockRockyApplication()
    study = app.CreateProject().GetStudy()
    surface = study.CreateCircularSurface()
    surface.SetMaxRadius(0.3, unit="m")
    particle = study.CreateParticle()

    inlet = study.CreateParticleInlet(entry_point=surface, particle=particle)
    inlet.GetInputPropertiesList()[0].SetMassFlowRate(1000.0, unit="t/h")
    first = study._derived()["velocity"]

    inlet.GetInputPropertiesList()[0].SetMassFlowRate(2000.0, unit="t/h")
    second = study._derived()["velocity"]

    assert second == pytest.approx(2.0 * first)
    assert first > 0.0
    assert math.isfinite(first)


def test_grid_function_reports_velocity() -> None:
    """The velocity grid function grows with the simulation time."""
    app = MockRockyApplication()
    study = app.CreateProject().GetStudy()
    surface = study.CreateCircularSurface()
    surface.SetMaxRadius(0.3, unit="m")
    particle = study.CreateParticle()
    particle.SetMinimumSize(0.05, unit="m")
    inlet = study.CreateParticleInlet(entry_point=surface, particle=particle)
    inlet.GetInputPropertiesList()[0].SetMassFlowRate(1000.0, unit="t/h")

    study.GetSolver().SetSimulationDuration(2.0, unit="s")
    study.StartSimulation(non_blocking=False)

    grid_function = study.GetParticles().GetGridFunction("Velocity : Translational : Absolute")
    times = study.GetTimeSet()
    first = grid_function.GetMax(unit="m/s", time_step=0)
    last = grid_function.GetMax(unit="m/s", time_step=len(times) - 1)
    assert last > first >= 0.0


def test_selection_process_filters_the_curve() -> None:
    """A selection returns a fraction of the domain curve."""
    app = MockRockyApplication()
    study = app.CreateProject().GetStudy()
    surface = study.CreateCircularSurface()
    particle = study.CreateParticle()
    inlet = study.CreateParticleInlet(entry_point=surface, particle=particle)
    inlet.GetInputPropertiesList()[0].SetMassFlowRate(1000.0, unit="t/h")
    study.GetSolver().SetSimulationDuration(2.0, unit="s")
    study.StartSimulation(non_blocking=False)

    processes = study.GetUserProcessCollection()
    selection = processes.CreateCylinderProcess(study.GetParticles())
    selection.SetSize(1.0, 1.0, 1.0, unit="m")
    selection.SetCenter(0.0, 0.0, 0.0, unit="m")

    _, whole = study.GetParticles().GetNumpyCurve("Particles Count")
    _, subset = selection.GetNumpyCurve("Particles Count")
    assert subset[-1] < whole[-1]


def test_save_project_writes_a_file(tmp_path) -> None:
    """Saving a project writes a readable file."""
    app = MockRockyApplication()
    project = app.CreateProject()
    target = tmp_path / "demo.rocky"
    project.SaveProject(str(target))

    assert target.exists()
    assert "Study" in target.read_text(encoding="utf-8")


def test_backend_reports_itself_as_simulated() -> None:
    """The mock backend is always tagged as simulated."""
    backend = MockBackend()
    status = backend.probe()
    assert status.simulated is True
    assert status.mode == "mock"
    assert backend.simulated is True
