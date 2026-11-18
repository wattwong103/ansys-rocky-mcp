"""Tests for the Rocky MCP tools, driven through an in-memory MCP client."""

from __future__ import annotations

import math

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


async def test_curve_names_can_be_listed(study: Client) -> None:
    """The domain publishes the curve names the reader accepts."""
    listed = await study.call_tool("rocky_list_curves")
    assert "Particles Count" in listed.data["curves"]
    assert "Particles Mass Flow In" in listed.data["curves"]


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
            "specific_heat": 800.0,
            "thermal_conductivity": 0.3,
            "particle_id": particles[0],
        },
    )
    assert created.data["name"] == "sand"
    assert created.data["density"] == pytest.approx(2600.0)
    assert created.data["youngs_modulus"] == pytest.approx(1.0e7)
    assert created.data["specific_heat_j_kg_k"] == pytest.approx(800.0)
    assert created.data["thermal_conductivity_w_m_k"] == pytest.approx(0.3)
    assert created.data["particle_id"] == particles[0]
    as_pascals = await study.call_tool(
        "rocky_create_material",
        {"name": "soft", "youngs_modulus": 2.0e6, "modulus_unit": "Pa"},
    )
    assert as_pascals.data["youngs_modulus"] == pytest.approx(2.0e6)
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
            "adhesive_distance": 1.0e-5,
            "tangential_stiffness_ratio": 0.8,
            "velocity_exponent": 0.5,
            "velocity_limit": 1.0,
            "contact_stiffness_multiplier": 2.0,
            "adhesive_fraction": 0.25,
            "surface_energy": 0.04,
        },
    )
    assert pair.data["static_friction"] == pytest.approx(0.6)
    assert pair.data["adhesive_distance_m"] == pytest.approx(1.0e-5)
    assert pair.data["tangential_stiffness_ratio"] == pytest.approx(0.8)
    assert pair.data["velocity_exponent"] == pytest.approx(0.5)
    assert pair.data["velocity_limit_m_s"] == pytest.approx(1.0)
    assert pair.data["contact_stiffness_multiplier"] == pytest.approx(2.0)
    assert pair.data["adhesive_fraction"] == pytest.approx(0.25)
    assert pair.data["surface_energy"] == pytest.approx(0.04)
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
    timed = await study.call_tool(
        "rocky_set_simulation_settings",
        {"fixed_timestep": 1.0e-5},
    )
    assert timed.data["timestep_model"] == "constant"
    assert timed.data["fixed_timestep_s"] == pytest.approx(1.0e-5)
    assert timed.data["use_fixed_timestep"] is True
    cores = await study.call_tool("rocky_set_simulation_settings", {"processors": 4})
    assert cores.data["processors"] == 4
    assert cores.data["local_processors"] == 4
    assert cores.data["mpi_processors"] == 4
    released = await study.call_tool(
        "rocky_set_simulation_settings",
        {"release_without_overlap_check": True, "breakage_delay": 0.05},
    )
    assert released.data["release_without_overlap_check"] is True
    assert released.data["breakage_delay_s"] == pytest.approx(0.05)
    listed = await study.call_tool("rocky_set_output_property", {})
    assert "Particles" in listed.data["available_standard"]
    output = await study.call_tool(
        "rocky_set_output_property",
        {"names": ["Particles", "Count"], "enabled": True},
    )
    assert output.data["enabled"] is True
    assert "Particles" in output.data["available"]
    gpu = await study.call_tool(
        "rocky_set_simulation_settings",
        {"processing_unit": "GPU", "target_gpus": [0], "gpu_slicing_direction": "Y"},
    )
    assert gpu.data["processing_unit"] == "GPU"
    assert gpu.data["local_processing_unit"] == "GPU"
    assert gpu.data["target_gpus"] == [0]
    assert gpu.data["gpu_slicing_direction"] == "Y"
    with pytest.raises(ToolError, match="processing unit"):
        await study.call_tool(
            "rocky_set_simulation_settings",
            {"processing_unit": "tpu"},
        )

    description = await study.call_tool("rocky_describe_study")
    assert description.data["study"]["domain"]["max_m"] == pytest.approx([5.0, 1.5, 1.5])
    assert description.data["study"]["solver"]["fixed_timestep_s"] == pytest.approx(1.0e-5)
    assert description.data["study"]["solver"]["processors"] == 4
    assert description.data["study"]["solver"]["release_without_overlap_check"] is True
    periodic = await study.call_tool(
        "rocky_set_domain",
        {"domain_type": "CARTESIAN", "periodic_directions": "XY"},
    )
    assert periodic.data["domain_type"] == "CARTESIAN"
    assert periodic.data["cartesian_periodic_directions"] == "XY"
    bounded = await study.call_tool(
        "rocky_set_domain",
        {
            "periodic_min": [-2.0, -0.5, 0.0],
            "periodic_max": [2.0, 0.5, 1.0],
            "periodic_at_geometry_limits": False,
        },
    )
    assert bounded.data["periodic_min_m"] == pytest.approx([-2.0, -0.5, 0.0])
    assert bounded.data["periodic_max_m"] == pytest.approx([2.0, 0.5, 1.0])
    assert bounded.data["periodic_at_geometry_limits"] is False
    described = await study.call_tool("rocky_describe_study")
    assert described.data["study"]["domain"]["periodic_min_m"] == pytest.approx([-2.0, -0.5, 0.0])
    assert described.data["study"]["domain"]["periodic_at_geometry_limits"] is False
    axis = await study.call_tool(
        "rocky_set_domain",
        {"domain_type": "CYLINDRICAL", "periodic_directions": "Z"},
    )
    assert axis.data["cylindrical_periodic_directions"] == "Z"
    divided = await study.call_tool(
        "rocky_set_domain",
        {"domain_type": "CYLINDRICAL", "number_of_divisions": 4},
    )
    assert divided.data["number_of_divisions"] == 4
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
            "restitution_model": "velocity_dependent",
            "thermal": True,
        },
    )
    assert models.data["rolling_resistance"] == "type_3"
    assert models.data["adhesion"] == "JKR"
    assert models.data["restitution_model"] == "velocity_dependent"
    assert models.data["thermal"] is True
    again = await study.call_tool("rocky_describe_study")
    assert again.data["study"]["contact_models"]["normal_force"] == "damped_hertzian"


async def test_a_feed_conveyor_records_belt_speed(study: Client) -> None:
    """A feed conveyor keeps the belt speed, width, and incline."""
    created = await study.call_tool(
        "rocky_create_feed_conveyor",
        {
            "belt_speed": 2.5,
            "belt_width": 0.8,
            "incline_deg": 12.0,
            "loading_length": 6.0,
            "drop_length": 1.2,
            "drop_width": 0.8,
            "drop_height": 0.5,
            "skirtboard_height": 0.3,
            "transition_length": 1.5,
            "idler_diameter": 0.4,
            "wall_thickness": 0.008,
            "face_width": 0.35,
            "alignment_deg": 15.0,
            "return_belt_deg": 12.0,
            "out_of_plane_offset": 0.15,
            "horizontal_offset": 1.25,
            "vertical_offset": 0.4,
            "name": "feed",
        },
    )
    assert created.data["name"] == "feed"
    assert created.data["loading_length_m"] == pytest.approx(6.0)
    assert created.data["drop_box_m"] == pytest.approx([1.2, 0.8, 0.5])
    assert created.data["skirtboard_height_m"] == pytest.approx(0.3)
    assert created.data["transition_length_m"] == pytest.approx(1.5)
    assert created.data["idler_diameter_m"] == pytest.approx(0.4)
    assert created.data["wall_thickness_m"] == pytest.approx(0.008)
    assert created.data["face_width_m"] == pytest.approx(0.35)
    assert created.data["belt_speed_m_s"] == pytest.approx(2.5)
    assert created.data["belt_width_m"] == pytest.approx(0.8)
    assert created.data["incline_deg"] == pytest.approx(12.0)
    assert created.data["alignment_deg"] == pytest.approx(15.0)
    assert created.data["return_belt_deg"] == pytest.approx(12.0)
    assert created.data["out_of_plane_offset_m"] == pytest.approx(0.15)
    assert created.data["horizontal_offset_m"] == pytest.approx(1.25)
    assert created.data["vertical_offset_m"] == pytest.approx(0.4)
    receiving = await study.call_tool(
        "rocky_create_receiving_conveyor",
        {
            "belt_speed": 1.2,
            "start_time": 3.0,
            "acceleration_period": 1.5,
            "stop_time": 20.0,
            "deceleration_period": 2.0,
            "belt_profile": "Five Rolls",
            "triangle_size": 0.02,
            "belt_thickness": 0.012,
            "alignment_deg": -8.0,
            "return_belt_deg": 5.0,
            "out_of_plane_offset": -0.05,
            "horizontal_offset": -0.5,
            "vertical_offset": 2.0,
            "name": "discharge",
        },
    )
    assert receiving.data["kind"] == "receiving"
    assert receiving.data["alignment_deg"] == pytest.approx(-8.0)
    assert receiving.data["return_belt_deg"] == pytest.approx(5.0)
    assert receiving.data["out_of_plane_offset_m"] == pytest.approx(-0.05)
    assert receiving.data["horizontal_offset_m"] == pytest.approx(-0.5)
    assert receiving.data["vertical_offset_m"] == pytest.approx(2.0)
    assert receiving.data["start_time_s"] == pytest.approx(3.0)
    assert receiving.data["acceleration_period_s"] == pytest.approx(1.5)
    assert receiving.data["stop_time_s"] == pytest.approx(20.0)
    assert receiving.data["deceleration_period_s"] == pytest.approx(2.0)
    assert receiving.data["belt_profile"] == "Five Rolls"
    assert receiving.data["triangle_size_m"] == pytest.approx(0.02)
    assert receiving.data["belt_thickness_m"] == pytest.approx(0.012)
    assert receiving.data["name"] == "discharge"
    assert receiving.data["belt_speed_m_s"] == pytest.approx(1.2)
    belt = await study.call_tool(
        "rocky_create_material",
        {"name": "belt", "density": 1200.0, "conveyor_id": created.data["conveyor_id"]},
    )
    assert belt.data["conveyor_id"] == created.data["conveyor_id"]


async def test_unknown_particle_shape_is_rejected(study: Client) -> None:
    """A shape Rocky does not define is rejected before the solver call."""
    with pytest.raises(ToolError, match="sphere"):
        await study.call_tool("rocky_create_particle", {"shape": "cube"})


async def test_particle_shape_dimensions_are_stored(study: Client) -> None:
    """Non-spherical shape dimensions use Rocky's meters and dega tokens."""
    created = await study.call_tool(
        "rocky_create_particle",
        {
            "shape": "faceted_cylinder",
            "horizontal_aspect_ratio": 1.5,
            "vertical_aspect_ratio": 2.0,
            "thickness": 0.01,
            "number_of_corners": 6,
            "edge_angle_deg": 30.0,
            "size_type": "equivalent_diameter",
            "rolling_resistance": 0.3,
            "breakage_model": "ab_t10",
            "abt10_reference_size": 0.05,
            "abt10_reference_energy": 40.0,
            "tavares_a": 60.0,
            "tavares_d0": 0.02,
            "tavares_e_inf": 100.0,
            "shear_stress_limit": 1.0e6,
            "stress_unit": "Pa",
            "plastic_limit": 2.0e6,
            "porosity": 0.2,
            "bending_angle_deg": 15.0,
            "torsion_angle_deg": 10.0,
            "elasticity": "anisotropic",
            "target_element_count": 8,
            "elastic_ratio_bending": 0.4,
            "elastic_ratio_torsion": 0.2,
            "element_damping_ratio": 0.1,
            "joint_damping_ratio": 0.05,
            "failure_ratio": 0.15,
            "name": "flake",
        },
    )
    assert created.data["name"] == "flake"
    assert created.data["horizontal_aspect_ratio"] == pytest.approx(1.5)
    assert created.data["vertical_aspect_ratio"] == pytest.approx(2.0)
    assert created.data["thickness_m"] == pytest.approx(0.01)
    assert created.data["number_of_corners"] == 6
    assert created.data["edge_angle_deg"] == pytest.approx(30.0)
    assert created.data["size_type"] == "equivalent_diameter"
    assert created.data["rolling_resistance"] == pytest.approx(0.3)
    assert created.data["breakage_model"] == "ab_t10"
    assert created.data["breakage_enabled"] is True
    assert created.data["abt10_reference_size_m"] == pytest.approx(0.05)
    assert created.data["abt10_reference_energy_j_kg"] == pytest.approx(40.0)
    assert created.data["tavares_a_percent"] == pytest.approx(60.0)
    assert created.data["tavares_d0_m"] == pytest.approx(0.02)
    assert created.data["tavares_e_inf_j_kg"] == pytest.approx(100.0)
    assert created.data["shear_stress_limit_n_m2"] == pytest.approx(1.0e6)
    assert created.data["plastic_limit_pa"] == pytest.approx(2.0e6)
    assert created.data["porosity"] == pytest.approx(0.2)
    assert created.data["bending_angle_deg"] == pytest.approx(15.0)
    assert created.data["torsion_angle_deg"] == pytest.approx(10.0)
    assert created.data["flexible"] is True
    assert created.data["elasticity"] == "anisotropic"
    assert created.data["target_element_count"] == 8
    assert created.data["elastic_ratio_bending"] == pytest.approx(0.4)
    assert created.data["elastic_ratio_torsion"] == pytest.approx(0.2)
    assert created.data["element_damping_ratio"] == pytest.approx(0.1)
    assert created.data["joint_damping_ratio"] == pytest.approx(0.05)
    assert created.data["failure_ratio"] == pytest.approx(0.15)
    assert created.data["with_failure"] is True
    oriented = await study.call_tool(
        "rocky_create_particle",
        {"shape": "sphere", "random_angle_deg": [10.0, 0.0, 5.0], "name": "tilted"},
    )
    assert oriented.data["random_angle"] is True
    assert oriented.data["random_angle_deg"] == pytest.approx([10.0, 0.0, 5.0])


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
        {
            "particle_id": particles[0],
            "model": "incomplete_beta",
            "minimum_size": 0.02,
            "minimum_size_ratio": 0.25,
            "surface_energy": 0.05,
        },
    )
    assert updated.data["model"] == "incomplete_beta"
    assert updated.data["minimum_size_m"] == pytest.approx(0.02)
    assert updated.data["minimum_size_ratio"] == pytest.approx(0.25)
    assert updated.data["surface_energy"] == pytest.approx(0.05)


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
            "start_time": 0.5,
            "stop_time": 4.0,
            "target_velocity": 1.2,
            "period": 3.0,
            "velocity_x": 0.2,
            "velocity_z": -0.1,
            "force_packing": True,
            "sph_temperature": 320.0,
        },
    )
    assert created.data["injection_duration_s"] == pytest.approx(1.5)
    assert created.data["start_time_s"] == pytest.approx(0.5)
    assert created.data["stop_time_s"] == pytest.approx(4.0)
    assert created.data["target_velocity_m_s"] == pytest.approx(1.2)
    assert created.data["periodic"] is True
    assert created.data["period_s"] == pytest.approx(3.0)
    assert created.data["velocity_x_m_s"] == pytest.approx(0.2)
    assert created.data["velocity_z_m_s"] == pytest.approx(-0.1)
    assert created.data["stop_all"] is True
    assert created.data["force_packing"] is True
    assert created.data["sph_temperature_k"] == pytest.approx(320.0)
    assert created.data["sph_injection"] is True
    extra = await study.call_tool(
        "rocky_create_particle",
        {"shape": "sphere", "name": "fines"},
    )
    added = await study.call_tool(
        "rocky_add_inlet_particle",
        {
            "inlet_id": created.data["inlet_id"],
            "particle_id": extra.data["particle_id"],
            "mass_flow_rate": 2.5,
        },
    )
    assert added.data["particle_count"] == 2
    removed = await study.call_tool(
        "rocky_remove_inlet_particle",
        {"inlet_id": created.data["inlet_id"], "particle_id": extra.data["particle_id"]},
    )
    assert removed.data["particle_count"] == 1
    assert added.data["unit"] == "t/h"
    heated = await study.call_tool(
        "rocky_set_temperature",
        {"object_id": created.data["inlet_id"], "temperature": 350.0},
    )
    assert heated.data["temperature_k"] == pytest.approx(350.0)
    assert heated.data["category"] == "inlet"

    with pytest.raises(ToolError, match="surface_id or conveyor_id"):
        await study.call_tool(
            "rocky_create_inlet",
            {"particle_id": particles[0], "mass_flow_rate": 10.0},
        )
    conveyor = await study.call_tool("rocky_create_feed_conveyor", {"name": "feed"})
    fed = await study.call_tool(
        "rocky_create_inlet",
        {
            "conveyor_id": conveyor.data["conveyor_id"],
            "particle_id": particles[0],
            "mass_flow_rate": 5.0,
        },
    )
    assert fed.data["conveyor_id"] == conveyor.data["conveyor_id"]
    assert fed.data["surface_id"] is None
    assert fed.data["inlet_id"]
    receiving = await study.call_tool("rocky_create_receiving_conveyor", {"name": "discharge"})
    with pytest.raises(ToolError, match="feed conveyor"):
        await study.call_tool(
            "rocky_create_inlet",
            {
                "conveyor_id": receiving.data["conveyor_id"],
                "particle_id": particles[0],
                "mass_flow_rate": 5.0,
            },
        )


async def test_volumetric_inlet_fills_a_box(study: Client) -> None:
    """A volumetric inlet can fill a box with a given particle mass."""
    objects = await study.call_tool("rocky_list_processes")
    particles = [
        entry["id"]
        for entry in objects.data["registered_objects"]
        if entry["category"] == "particle"
    ]
    with pytest.raises(ToolError, match="three numbers"):
        await study.call_tool(
            "rocky_create_volumetric_inlet",
            {
                "particle_id": particles[0],
                "box_center": [0.0, 0.0],
                "box_dimensions": [1.0, 1.0, 1.0],
            },
        )
    created = await study.call_tool(
        "rocky_create_volumetric_inlet",
        {
            "particle_id": particles[0],
            "name": "hopper fill",
            "mass": 25.0,
            "box_center": [0.0, 0.0, 0.5],
            "box_dimensions": [1.0, 0.4, 0.8],
        },
    )
    assert created.data["name"] == "hopper fill"
    assert created.data["mass_kg"] == pytest.approx(25.0)
    assert created.data["fill"] == "box"


async def test_volumetric_inlet_fills_selected_walls(study: Client, tmp_path) -> None:
    """A volumetric inlet can fill the interior of selected walls."""
    geometry = tmp_path / "bin.stl"
    geometry.write_text("solid bin\nendsolid bin\n", encoding="utf-8")
    imported = await study.call_tool("rocky_import_wall", {"path": str(geometry)})
    objects = await study.call_tool("rocky_list_processes")
    particles = [
        entry["id"]
        for entry in objects.data["registered_objects"]
        if entry["category"] == "particle"
    ]
    with pytest.raises(ToolError, match="geometry id"):
        await study.call_tool(
            "rocky_create_volumetric_inlet",
            {"particle_id": particles[0], "use_geometries_to_compute": True},
        )
    created = await study.call_tool(
        "rocky_create_volumetric_inlet",
        {
            "particle_id": particles[0],
            "mass": 40.0,
            "use_geometries_to_compute": True,
            "geometry_ids": imported.data["wall_ids"],
        },
    )
    assert created.data["fill"] == "geometries"
    assert created.data["mass_kg"] == pytest.approx(40.0)


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
        {
            "kind": "rotation",
            "targets": [surfaces[0]],
            "angular_velocity": [0.0, 0.5, 0.0],
            "period": 2.0,
        },
    )
    assert motion.data["period_s"] == pytest.approx(2.0)
    assert motion.data["angular_velocity_rad_s"] == pytest.approx([0.0, 0.5, 0.0])
    in_degrees = await study.call_tool(
        "rocky_set_motion",
        {
            "kind": "rotation",
            "targets": [surfaces[0]],
            "angular_velocity": [0.0, 180.0, 0.0],
            "unit": "deg/s",
            "angular_acceleration": [0.0, 180.0, 0.0],
            "acceleration_unit": "deg/s2",
        },
    )
    assert in_degrees.data["angular_velocity_rad_s"] == pytest.approx([0.0, math.pi, 0.0])
    assert in_degrees.data["angular_acceleration_rad_s2"] == pytest.approx([0.0, math.pi, 0.0])
    sliding = await study.call_tool(
        "rocky_set_motion",
        {
            "kind": "translation",
            "targets": [surfaces[0]],
            "linear_velocity": [0.1, 0.0, 0.0],
            "linear_acceleration": [0.0, 0.0, -9.81],
        },
    )
    assert sliding.data["linear_acceleration_m_s2"] == pytest.approx([0.0, 0.0, -9.81])
    assert sliding.data["define_as"] == "fixed_acceleration"
    coasting = await study.call_tool(
        "rocky_set_motion",
        {
            "kind": "translation",
            "targets": [surfaces[0]],
            "linear_velocity": [1.0, 0.0, 0.0],
            "final_velocity": [0.0, 0.0, 0.0],
        },
    )
    assert coasting.data["define_as"] == "initial_and_final_velocities"
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

    result = await study.call_tool(
        "rocky_import_wall",
        {
            "path": str(geometry),
            "convert_yz": True,
            "geometry_split_method": "PARTS",
            "normal_angle_tolerance": 12.0,
            "max_edge_size": 0.05,
            "triangle_size": 0.01,
            "translation": [1.0, 0.0, 0.5],
            "rotation_deg": [0.0, 90.0, 0.0],
            "periodic_replication": True,
        },
    )
    assert result.data["wall_ids"]
    assert result.data["convert_yz"] is True
    assert result.data["geometry_split_method"] == "PARTS"
    assert result.data["normal_angle_tolerance"] == pytest.approx(12.0)
    assert result.data["max_edge_size_m"] == pytest.approx(0.05)
    assert result.data["triangle_size_m"] == pytest.approx(0.01)
    assert result.data["translation_m"] == pytest.approx([1.0, 0.0, 0.5])
    assert result.data["rotation_deg"] == pytest.approx([0.0, 90.0, 0.0])
    assert result.data["periodic_replication"] is True
    boundary = await study.call_tool(
        "rocky_set_sph_boundary",
        {
            "wall_id": result.data["wall_ids"][0],
            "boundary_type": "no_slip_turbulent",
            "contact_angle_deg": 30.0,
            "capillary_friction": 0.3,
        },
    )
    assert boundary.data["boundary_type"] == "no_slip_turbulent"
    assert boundary.data["contact_angle_deg"] == pytest.approx(30.0)
    assert boundary.data["capillary_friction"] == pytest.approx(0.3)
    wear = await study.call_tool(
        "rocky_set_wall_wear",
        {"wall_id": result.data["wall_ids"][0], "volume_shear_work_ratio": 1.0e-12},
    )
    assert wear.data["wear_model"] == "archard"
    assert wear.data["volume_shear_work_ratio_m3_j"] == pytest.approx(1.0e-12)
    heated_wall = await study.call_tool(
        "rocky_set_temperature",
        {"object_id": result.data["wall_ids"][0], "temperature": 350.0},
    )
    assert heated_wall.data["temperature_k"] == pytest.approx(350.0)
    assert heated_wall.data["thermal_boundary"] == "prescribed_temperature"
    heavy = await study.call_tool(
        "rocky_set_wall_mass",
        {
            "wall_id": result.data["wall_ids"][0],
            "mass": 12.0,
            "inertia": [0.2, 0.3, 0.4],
            "disable_time": 5.0,
        },
    )
    assert heavy.data["mass_kg"] == pytest.approx(12.0)
    assert heavy.data["inertia_kg_m2"] == pytest.approx([0.2, 0.3, 0.4])
    assert heavy.data["disable_time_s"] == pytest.approx(5.0)
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
