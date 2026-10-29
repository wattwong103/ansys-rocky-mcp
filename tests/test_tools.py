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


async def test_domain_and_solver_settings_are_applied(study: Client) -> None:
    """Domain and solver settings are visible in the study description."""
    await study.call_tool(
        "rocky_set_domain", {"min_values": [-5.0, -1.5, -1.5], "max_values": [5.0, 1.5, 1.5]}
    )
    await study.call_tool("rocky_set_simulation_settings", {"duration": 4.0, "time_interval": 0.2})

    description = await study.call_tool("rocky_describe_study")
    assert description.data["study"]["domain"]["max_m"] == pytest.approx([5.0, 1.5, 1.5])
    assert description.data["study"]["solver"]["duration_s"] == pytest.approx(4.0)
    assert description.data["study"]["solver"]["time_interval_s"] == pytest.approx(0.2)


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


async def test_disconnect_closes_the_session(client: Client) -> None:
    """Disconnecting reports that the session was closed."""
    await client.call_tool("rocky_create_project")
    result = await client.call_tool("rocky_disconnect")
    assert result.data["disconnected"] is True
