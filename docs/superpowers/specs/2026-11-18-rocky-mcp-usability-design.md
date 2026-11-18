# Make ansys-rocky-mcp usable with real Rocky

## Scope

The existing MCP tools must be able to complete the official PyRocky minimal example against a real Ansys Rocky session:

1. `CreateProject`
2. `CreateParticle`
3. `CreateCircularSurface`
4. `CreateParticleInlet`
5. Domain limits
6. `SetSimulationDuration`
7. `SaveProject`
8. `StartSimulation`
9. `GetParticles().GetNumpyCurve("Particles Count")`

Correct the tool and backend calls that the current Rocky stubs reject. This slice does not add materials, contacts, gravity, export, or volumetric inlets.

## Architecture

Keep the shared application object. Tools keep calling the Rocky application. The mock keeps implementing that same object. Fix the calls whose arguments do not match the stubs, and add a live test for the minimal sequence when Rocky is installed.

Do not add a study facade. The application object is already the shared interface, and only a handful of arguments are wrong.

## Corrections

Connect uses the `api` property. Installed `RockyClient.api` is a property that returns the application proxy. `PyRockyBackend` currently calls `self._client.api()`, which fails before `CreateProject`. The official example uses `rocky.api.CreateProject()`. `connect` reads `client.api` once and stores that object. A missing or failed launch still raises `RockyUnavailableError` with the install and license hint. Attach still uses `connect(port=...)`.

Launch uses the cold-start wait. `launch_rocky` treats a missing `connect_timeout` as its own longer wait. The backend always passes `connect_timeout` from config, and that default is 30 seconds. `launch_timeout` (default 300 seconds) is stored and never passed. On launch, pass `connect_timeout` from `launch_timeout`. `ROCKY_MCP_LAUNCH_TIMEOUT` is the knob for a slow Rocky start. The 30 second `connect_timeout` field stays in config and is not the cold-start wait.

`rocky_set_simulation_settings` accepts only `local`, `single_mpi`, and `multi_mpi`. Any other value, including `serial` and `parallel`, raises `ToolError` before the Rocky call. The message names the three legal values. Omitting `execution_mode` leaves Rocky's current mode unchanged. The mock solver default becomes `local`. `GetValidExecutionModeValues` returns those three strings.

Selection creation passes `center`, `size`, and `final_angle` into `CreateCylinderProcess`, and `center` plus `magnitude` into `CreateCubeProcess`. The tools do not call `SetSize` or `SetCenter` afterward. Omitted size or center is omitted on the constructor. A provided size or center must be three numbers; otherwise the tool raises `ToolError` before the Rocky call and does not pad a short vector. The mock's `CreateCubeProcess` accepts `magnitude` instead of `size`, so the tool and the mock share one call. `final_angle` applies only to a cylinder.

Motion already matches the official motion example (`GetMotionFrameSource().NewFrame()`, `AddRotationMotion`, `ApplyTo`). Leave it.

README and the backend docstring say `RockyClient.api`, not `RockyClient.api()`.

## Error handling

Failed launch and a missing install keep today's `RockyUnavailableError` text. Execution-mode and selection-shape errors are `ToolError` and do not start a Rocky call. The mock rejects the same illegal execution modes, so those tests do not need a solver.

## Testing

`tests/test_pyrocky_backend.py` uses a fake client whose `api` is a property. Connect stores that object and never calls it. The same file patches `launch_rocky` and asserts the default config passes `connect_timeout=300`, not 30. No Rocky process.

`tests/test_tools.py`: `execution_mode="parallel"` raises `ToolError` naming `local`. `execution_mode="local"` is accepted. `test_selection_process_filters_results` keeps its MCP arguments (`size`, `center`) and still asserts the curve shrinks.

`tests/test_mock_backend.py`: `test_selection_process_filters_the_curve` passes `size` and `center` into `CreateCylinderProcess`. A cube created with `magnitude` stores that size. The rest of the mock suite stays green. No current test asserts the string `parallel`.

`tests/test_live_rocky.py` is skipped when no Rocky 2024 R2 or newer install is detected. With `ROCKY_MCP_LIVE=1`, a missing install fails the test. When Rocky is present, the test runs the minimal sequence headless through the MCP tools, asserts `simulated` is false, and asserts the Particles Count curve has at least one point.

This environment has no Rocky install, so the live path cannot be executed here. The skip, and the failure under `ROCKY_MCP_LIVE=1`, can.

## Out of scope

Materials, contact physics, gravity, result export, volumetric inlets, and wiring `rocky_run_python` to the connected study. The escape hatch stays a separate interpreter.
