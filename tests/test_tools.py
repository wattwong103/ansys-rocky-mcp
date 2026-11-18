"""Tests for the Rocky MCP tools, driven through an in-memory MCP client."""

from __future__ import annotations

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError


async def test_status_reports_the_backend(client: Client) -> None:
    """``rocky_status`` reports the mock backend as simulated."""
    result = await client.call_tool("rocky_status")
    assert result.data["backend"] == "mock"
    assert result.data["simulated"] is True
    assert result.data["config"]["backend"] == "mock"


async def test_tools_requiring_a_project_fail_clearly(client: Client) -> None:
    """A setup tool without an open project reports what to do next."""
    with pytest.raises(ToolError, match="rocky_create_project"):
        await client.call_tool("rocky_create_surface", {"kind": "circular"})


async def test_full_workflow(study: Client) -> None:
    """The complete build-run-read workflow succeeds."""
    description = await study.call_tool("rocky_describe_study")
    assert description.data["simulated"] is True
    classes = {element["class"] for element in description.data["study"]["elements"]}
    assert "RAParticle" in classes
    assert "RACircularSurface" in classes
    assert "RAParticleInlet" in classes

    started = await study.call_tool("rocky_run_simulation", {"non_blocking": False})
    assert started.data["progress_percent"] == pytest.approx(100.0)

    status = await study.call_tool("rocky_simulation_status")
    assert status.data["simulating"] is False
    assert status.data["has_results"] is True
    assert status.data["number_of_time_steps"] == pytest.approx(21.0)

    curve = await study.call_tool("rocky_get_curve", {"name": "Particles Count"})
    assert curve.data["number_of_points"] == 21
    assert curve.data["values"][-1] > curve.data["values"][0]
    assert len(curve.data["times"]) == 21

    flow = await study.call_tool(
        "rocky_get_curve", {"name": "Particles Mass Flow In", "unit": "t/h"}
    )
    assert flow.data["values"][-1] == pytest.approx(1000.0)


async def test_a_running_simulation_can_be_stopped(study: Client) -> None:
    """Stopping a non-blocking run clears the simulating flag."""
    await study.call_tool("rocky_set_simulation_settings", {"duration": 30.0})
    await study.call_tool("rocky_run_simulation", {"non_blocking": True})
    stopped = await study.call_tool("rocky_stop_simulation")
    assert stopped.data["simulating"] is False


async def test_a_finished_run_can_be_extended(study: Client) -> None:
    """Extending a finished run increases the duration and the time set."""
    await study.call_tool("rocky_run_simulation", {"non_blocking": False})
    before = await study.call_tool("rocky_describe_study")
    duration = before.data["study"]["solver"]["duration_s"]
    extended = await study.call_tool("rocky_extend_simulation", {"seconds": 1.0})
    assert extended.data["duration_s"] == pytest.approx(duration + 1.0)
    curve = await study.call_tool("rocky_get_curve", {"name": "Particles Count"})
    assert curve.data["times"][-1] == pytest.approx(duration + 1.0)


async def test_results_can_be_deleted(study: Client) -> None:
    """Deleting results clears the finished run."""
    await study.call_tool("rocky_run_simulation", {"non_blocking": False})
    deleted = await study.call_tool("rocky_delete_results")
    assert deleted.data["has_results"] is False
    status = await study.call_tool("rocky_simulation_status")
    assert status.data["has_results"] is False


async def test_a_curve_can_be_saved_as_csv(study: Client, tmp_path) -> None:
    """A curve can be written to CSV after the run."""
    await study.call_tool("rocky_run_simulation", {"non_blocking": False})
    destination = tmp_path / "count.csv"
    saved = await study.call_tool(
        "rocky_save_curve",
        {"name": "Particles Count", "path": str(destination)},
    )
    assert saved.data["number_of_points"] > 1
    text = destination.read_text(encoding="utf-8")
    assert text.startswith("time_s,value\n")


async def test_curves_are_requested_before_the_run(study: Client) -> None:
    """Requesting a curve before the simulation reports the missing step."""
    with pytest.raises(ToolError, match="rocky_run_simulation"):
        await study.call_tool("rocky_get_curve", {"name": "Particles Count"})


async def test_unknown_curve_lists_the_available_ones(study: Client) -> None:
    """An unknown curve name produces a helpful error."""
    await study.call_tool("rocky_run_simulation", {"non_blocking": False})
    with pytest.raises(ToolError, match="Particles Count"):
        await study.call_tool("rocky_get_curve", {"name": "Not A Curve"})


async def test_unknown_object_id_is_reported(study: Client) -> None:
    """Referring to an unknown object id lists the known ones."""
    with pytest.raises(ToolError, match="Known ids"):
        await study.call_tool(
            "rocky_create_inlet",
            {"surface_id": "surface_99", "particle_id": "particle_1", "mass_flow_rate": 10},
        )


async def test_grid_function_statistics(study: Client) -> None:
    """The grid function statistics span the whole time set."""
    await study.call_tool("rocky_run_simulation", {"non_blocking": False})

    names = await study.call_tool("rocky_list_grid_functions")
    assert "Velocity : Translational : Absolute" in names.data["grid_functions"]

    stats = await study.call_tool(
        "rocky_get_grid_function_stats", {"name": "Velocity : Translational : Absolute"}
    )
    assert stats.data["number_of_time_steps"] == 21
    assert stats.data["overall_max"] >= stats.data["overall_min"]
    assert len(stats.data["series"]) == 21


async def test_selection_process_filters_results(study: Client) -> None:
    """A selection returns a subset of the domain curve."""
    await study.call_tool("rocky_run_simulation", {"non_blocking": False})

    selection = await study.call_tool(
        "rocky_create_selection",
        {"kind": "cylinder", "size": [1.0, 1.0, 1.0], "center": [0.0, 0.0, 0.0]},
    )
    selection_id = selection.data["selection_id"]

    whole = await study.call_tool("rocky_get_curve", {"name": "Particles Count"})
    subset = await study.call_tool(
        "rocky_get_curve", {"name": "Particles Count", "selection_id": selection_id}
    )
    assert subset.data["values"][-1] < whole.data["values"][-1]

    processes = await study.call_tool("rocky_list_processes")
    assert selection.data["name"] in processes.data["processes"]


async def test_material_can_be_assigned_to_a_particle(study: Client) -> None:
    """A solid material is created and assigned to a particle."""
    objects = await study.call_tool("rocky_list_processes")
    particles = [
        entry["id"]
        for entry in objects.data["registered_objects"]
        if entry["category"] == "particle"
    ]
    created = await study.call_tool(
        "rocky_create_material",
        {
            "name": "sand",
            "density": 2600.0,
            "youngs_modulus": 1.0e7,
            "poisson_ratio": 0.3,
            "particle_id": particles[0],
        },
    )
    assert created.data["name"] == "sand"
    assert created.data["density"] == pytest.approx(2600.0)
    assert created.data["particle_id"] == particles[0]
    assert created.data["wall_id"] is None

    other = await study.call_tool("rocky_create_material", {"name": "steel", "density": 7800.0})
    pair = await study.call_tool(
        "rocky_set_material_interaction",
        {
            "material_a": created.data["material_id"],
            "material_b": other.data["material_id"],
            "static_friction": 0.6,
            "dynamic_friction": 0.4,
            "restitution": 0.2,
        },
    )
    assert pair.data["static_friction"] == pytest.approx(0.6)
    assert pair.data["dynamic_friction"] == pytest.approx(0.4)
    assert pair.data["restitution"] == pytest.approx(0.2)


async def test_material_can_be_assigned_to_a_wall(study: Client, tmp_path) -> None:
    """A solid material can be assigned to an imported wall."""
    geometry = tmp_path / "chute.stl"
    geometry.write_text("solid chute\nendsolid chute\n", encoding="utf-8")
    imported = await study.call_tool("rocky_import_wall", {"path": str(geometry)})
    created = await study.call_tool(
        "rocky_create_material",
        {"name": "rubber", "density": 1100.0, "wall_id": imported.data["wall_ids"][0]},
    )
    assert created.data["wall_id"] == imported.data["wall_ids"][0]
    assert created.data["name"] == "rubber"


async def test_walls_can_be_exported_to_stl(study: Client, tmp_path) -> None:
    """Exporting a wall writes an STL through the export toolkit."""
    geometry = tmp_path / "chute.stl"
    geometry.write_text("solid chute\nendsolid chute\n", encoding="utf-8")
    imported = await study.call_tool("rocky_import_wall", {"path": str(geometry)})
    destination = tmp_path / "out.stl"
    exported = await study.call_tool(
        "rocky_export_stl",
        {"path": str(destination), "wall_ids": imported.data["wall_ids"]},
    )
    assert exported.data["simulated"] is True
    assert destination.exists()
    assert "mock" in destination.read_text(encoding="utf-8")


async def test_domain_and_solver_settings_are_applied(study: Client) -> None:
    """Domain and solver settings are visible in the study description."""
    await study.call_tool(
        "rocky_set_domain", {"min_values": [-5.0, -1.5, -1.5], "max_values": [5.0, 1.5, 1.5]}
    )
    await study.call_tool("rocky_set_simulation_settings", {"duration": 4.0, "time_interval": 0.2})

    description = await study.call_tool("rocky_describe_study")
    assert description.data["study"]["domain"]["max_m"] == pytest.approx([5.0, 1.5, 1.5])
    periodic = await study.call_tool(
        "rocky_set_domain",
        {"domain_type": "CARTESIAN", "periodic_directions": "XY"},
    )
    assert periodic.data["domain_type"] == "CARTESIAN"
    assert periodic.data["cartesian_periodic_directions"] == "XY"
    axis = await study.call_tool(
        "rocky_set_domain",
        {"domain_type": "CYLINDRICAL", "periodic_directions": "Z"},
    )
    assert axis.data["cylindrical_periodic_directions"] == "Z"
    assert description.data["study"]["solver"]["duration_s"] == pytest.approx(4.0)
    assert description.data["study"]["solver"]["time_interval_s"] == pytest.approx(0.2)
    updated = await study.call_tool("rocky_set_simulation_settings", {"curves_frequency": 5})
    assert updated.data["curves_frequency"] == 5
    described = await study.call_tool("rocky_describe_study")
    assert described.data["study"]["solver"]["curves_frequency"] == 5

    gravity = await study.call_tool("rocky_set_gravity", {"gravity": [0.0, -9.81, 0.0]})
    assert gravity.data["gravity_m_s2"] == pytest.approx([0.0, -9.81, 0.0])
    described = await study.call_tool("rocky_describe_study")
    assert described.data["study"]["gravity_m_s2"] == pytest.approx([0.0, -9.81, 0.0])

    models = await study.call_tool(
        "rocky_set_contact_models",
        {
            "normal_force": "damped_hertzian",
            "tangential_force": "mindlin_deresiewicz",
            "rolling_resistance": "type_3",
            "adhesion": "JKR",
        },
    )
    assert models.data["rolling_resistance"] == "type_3"
    assert models.data["adhesion"] == "JKR"
    again = await study.call_tool("rocky_describe_study")
    assert again.data["study"]["contact_models"]["normal_force"] == "damped_hertzian"


async def test_particle_size_distribution_is_set(study: Client) -> None:
    """A particle can use one of Rocky's size distribution models."""
    objects = await study.call_tool("rocky_list_processes")
    particles = [
        entry["id"]
        for entry in objects.data["registered_objects"]
        if entry["category"] == "particle"
    ]
    updated = await study.call_tool(
        "rocky_set_size_distribution",
        {"particle_id": particles[0], "model": "incomplete_beta", "minimum_size": 0.02},
    )
    assert updated.data["model"] == "incomplete_beta"
    assert updated.data["minimum_size_m"] == pytest.approx(0.02)


async def test_inlet_injection_duration_is_stored(study: Client) -> None:
    """An inlet can stop feeding after a set duration."""
    objects = await study.call_tool("rocky_list_processes")
    surfaces = [
        entry["id"]
        for entry in objects.data["registered_objects"]
        if entry["category"] == "surface"
    ]
    particles = [
        entry["id"]
        for entry in objects.data["registered_objects"]
        if entry["category"] == "particle"
    ]
    created = await study.call_tool(
        "rocky_create_inlet",
        {
            "surface_id": surfaces[0],
            "particle_id": particles[0],
            "mass_flow_rate": 10.0,
            "injection_duration": 1.5,
        },
    )
    assert created.data["injection_duration_s"] == pytest.approx(1.5)


async def test_motion_is_applied_to_the_targets(study: Client) -> None:
    """A rotation motion is attached to the requested surface."""
    objects = await study.call_tool("rocky_list_processes")
    surfaces = [
        entry["id"]
        for entry in objects.data["registered_objects"]
        if entry["category"] == "surface"
    ]
    assert surfaces

    motion = await study.call_tool(
        "rocky_set_motion",
        {"kind": "rotation", "targets": [surfaces[0]], "angular_velocity": [0.0, 0.5, 0.0]},
    )
    assert motion.data["kind"] == "rotation"
    assert motion.data["targets"] == [surfaces[0]]


async def test_motion_requires_a_velocity(study: Client) -> None:
    """A rotation without an angular velocity is rejected."""
    with pytest.raises(ToolError, match="angular_velocity"):
        await study.call_tool("rocky_set_motion", {"kind": "rotation", "targets": ["surface_1"]})


async def test_wait_for_completion_without_a_running_job(study: Client) -> None:
    """Waiting when nothing runs returns immediately."""
    result = await study.call_tool("rocky_wait_for_completion")
    assert result.data["completed"] is True
    assert result.data["timed_out"] is False


async def test_save_and_reopen_a_project(study: Client, tmp_path) -> None:
    """A saved project can be reopened."""
    target = tmp_path / "study.rocky"
    saved = await study.call_tool("rocky_save_project", {"path": str(target)})
    assert saved.data["project_filename"] == str(target)
    assert target.exists()

    reopened = await study.call_tool("rocky_open_project", {"path": str(target)})
    assert reopened.data["project_filename"] == str(target)


async def test_open_project_reports_a_missing_file(client: Client) -> None:
    """Opening a missing project names the missing file."""
    with pytest.raises(ToolError, match="not found"):
        await client.call_tool("rocky_open_project", {"path": "does_not_exist.rocky"})


async def test_import_wall_reports_a_missing_file(client: Client) -> None:
    """Importing a missing geometry names the missing file."""
    await client.call_tool("rocky_create_project")
    with pytest.raises(ToolError, match="not found"):
        await client.call_tool("rocky_import_wall", {"path": "missing.stl"})


async def test_import_wall_creates_identified_walls(study: Client, tmp_path) -> None:
    """An imported wall is registered and can be referenced."""
    geometry = tmp_path / "wall.stl"
    geometry.write_text("solid wall\nendsolid wall\n", encoding="utf-8")

    result = await study.call_tool("rocky_import_wall", {"path": str(geometry)})
    assert result.data["wall_ids"]
    assert result.data["walls"][0]["id"] == result.data["wall_ids"][0]


async def test_parallel_execution_mode_is_rejected(study: Client) -> None:
    """``parallel`` is not a Rocky execution mode."""
    with pytest.raises(ToolError, match="local"):
        await study.call_tool("rocky_set_simulation_settings", {"execution_mode": "parallel"})


async def test_serial_execution_mode_is_rejected(study: Client) -> None:
    """``serial`` is not a Rocky execution mode."""
    with pytest.raises(ToolError, match="single_mpi"):
        await study.call_tool("rocky_set_simulation_settings", {"execution_mode": "serial"})


async def test_local_execution_mode_is_accepted(study: Client) -> None:
    """``local`` is stored as the solver execution mode."""
    result = await study.call_tool("rocky_set_simulation_settings", {"execution_mode": "local"})
    assert result.data["execution_mode"] == "local"


async def test_selection_size_requires_three_numbers(study: Client) -> None:
    """A short selection size is rejected and is not padded."""
    with pytest.raises(ToolError, match="three numbers"):
        await study.call_tool(
            "rocky_create_selection",
            {"kind": "cylinder", "size": [1.0], "center": [0.0, 0.0, 0.0]},
        )


async def test_disconnect_closes_the_session(client: Client) -> None:
    """Disconnecting reports that the session was closed."""
    await client.call_tool("rocky_create_project")
    result = await client.call_tool("rocky_disconnect")
    assert result.data["disconnected"] is True
