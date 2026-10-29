# ansys-rocky-mcp

**Model Context Protocol (MCP) server for Ansys Rocky** — the discrete element method (DEM)
solver. Drive Rocky studies, run simulations and post-process results from Claude, MiniMax
Code, Codex, Cursor, Kimi or any other MCP client.

Built on the official [`ansys-common-mcp`](https://common-mcp.docs.pyansys.com/) foundation,
so it follows the same extension pattern as the other PyAnsys MCP servers
(`pymapdl-mcp`, `pyfluent-mcp`, `pymechanical-mcp`, …).

---

## Why this exists

Ansys ships first-party MCP servers for Fluent, Mechanical, MAPDL, AEDT, CFX and Lumerical.
**There is no MCP server for Rocky.** The only Rocky automation that exists today is the
Python client library [`ansys-rocky-core`](https://pypi.org/project/ansys-rocky-core/),
which is an API, not an MCP server — so an AI assistant cannot call it without the user
writing the PyRocky boilerplate by hand every time.

This project closes that gap: a stateful, tool-based MCP server that lets an AI build a
Rocky study, run it, and read the results.

---

## Status

| | |
|---|---|
| **Tools** | 25 (session, project, setup, run control, results, Python escape hatch) |
| **Transport** | `stdio` (default) and `http` |
| **Verified** | Full workflow end-to-end over a real MCP stdio handshake, 42 automated tests |
| **Requires** | Python ≥ 3.12, `ansys-common-mcp`, `ansys-rocky-core` |
| **Real solver** | Needs a locally installed + licensed **Ansys Rocky 2024 R2 or newer** |

### ⚠️ Read this before trusting numbers

The server has two backends:

- **`pyrocky`** — drives a real Ansys Rocky application through PyRocky.
- **`mock`** — a built-in in-memory Rocky simulator. It models the study bookkeeping
  (project, particles, surfaces, inlets, outlets, selections, domain and solver settings)
  and produces **physically consistent but simulated** time series.

**Every response carries `backend` and `simulated` fields.** When `simulated` is `true`,
the numbers did **not** come from Ansys Rocky. The server instructions tell the AI to say
so explicitly.

The default backend is `auto`, which uses the real Rocky when it is installed and
**fails with an actionable error when it is not**. It never silently substitutes simulated
numbers for solver output.

---

## Install

```bash
git clone https://github.com/ansys/ansys-rocky-mcp.git
cd ansys-rocky-mcp

python -m venv .venv
.venv/Scripts/pip install -e .          # Windows
# .venv/bin/pip install -e .           # Linux / macOS
```

`ansys-common-mcp` requires Python ≥ 3.12. On Windows, `pip install` of the Ansys
namespace packages can conflict with a system-wide `ansys` install — use a dedicated
virtual environment as above.

---

## Quick start

### Try it without Ansys installed

```bash
.venv/Scripts/python.exe -m ansys_rocky_mcp --backend mock
```

### Drive a real Rocky installation

```bash
.venv/Scripts/python.exe -m ansys_rocky_mcp --backend auto --headless
```

Requires Ansys Rocky 2024 R2 or newer with `AWP_ROOT242` (or newer) set.

### As an MCP server over HTTP

```bash
.venv/Scripts/python.exe -m ansys_rocky_mcp --backend mock --transport http --http-port 8080
```

---

## Client setup

The server speaks standard MCP over `stdio`, so it works with any MCP client. Full
per-client instructions are in [`docs/mcp-clients.md`](docs/mcp-clients.md).

### Claude Desktop

`%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ansys-rocky": {
      "command": "N:\\ansys-rocky-mcp\\.venv\\Scripts\\python.exe",
      "args": ["-m", "ansys_rocky_mcp", "--backend", "mock"]
    }
  }
}
```

### Claude Code

```bash
claude mcp add --scope user ansys-rocky "N:\ansys-rocky-mcp\.venv\Scripts\python.exe" -m ansys_rocky_mcp --backend mock
```

### MiniMax Code

```json
{
  "mcpServers": {
    "ansys-rocky": {
      "type": "stdio",
      "command": "N:\\ansys-rocky-mcp\\.venv\\Scripts\\python.exe",
      "args": ["-m", "ansys_rocky_mcp", "--backend", "mock"]
    }
  }
}
```

### Codex CLI

`~/.codex/config.toml`:

```toml
[mcp_servers.ansys-rocky]
command = "N:\\ansys-rocky-mcp\\.venv\\Scripts\\python.exe"
args = ["-m", "ansys_rocky_mcp", "--backend", "mock"]
startup_timeout_sec = 90
```

---

## Tool surface

### Session

| Tool | Purpose |
|---|---|
| `rocky_status` | Backend, Rocky availability, detected versions, project and simulation state. Never launches Rocky. |
| `rocky_connect` | Launch or attach to Rocky (`auto` / `launch` / `attach`). |
| `rocky_disconnect` | Close the session. |

### Project

| Tool | Purpose |
|---|---|
| `rocky_create_project` | Create an empty project. |
| `rocky_open_project` | Open a `.rocky` project file. |
| `rocky_save_project` | Save the project. |
| `rocky_close_project` | Close the project. |
| `rocky_describe_study` | Full snapshot: elements, domain, solver, processes, simulation state. |

### Study setup

| Tool | Purpose |
|---|---|
| `rocky_import_wall` | Import a geometry file as a wall. |
| `rocky_create_surface` | Create a circular or rectangular surface. |
| `rocky_create_particle` | Create a particle definition (shape, sieve size). |
| `rocky_create_inlet` | Create a particle inlet and set its mass flow rate. |
| `rocky_create_outlet` | Create an outlet on a surface. |
| `rocky_set_domain` | Set the domain limits and boundary-limit behaviour. |
| `rocky_set_simulation_settings` | Duration, output interval, processors, execution mode. |
| `rocky_set_motion` | Apply a rotation or translation motion to objects. |

### Run control

| Tool | Purpose |
|---|---|
| `rocky_run_simulation` | Start the solver (blocking or non-blocking). |
| `rocky_simulation_status` | Progress, time set, results availability. |
| `rocky_wait_for_completion` | Wait for the run to finish. |

### Results

| Tool | Purpose |
|---|---|
| `rocky_get_curve` | Read a time series (`Particles Count`, `Particles Mass Flow In/Out`, …). |
| `rocky_list_grid_functions` | List the available grid functions. |
| `rocky_get_grid_function_stats` | Min/max of a grid function over the time set. |
| `rocky_create_selection` | Create a cylinder or cube selection process. |
| `rocky_list_processes` | List user processes and registered objects. |
| `rocky_run_python` | Run arbitrary Python in a persistent session (escape hatch). |

### Object identifiers

The Rocky API returns opaque proxy objects that an AI cannot reference later. This server
assigns each created object a short, stable identifier — `surface_1`, `particle_1`,
`inlet_1`, `outlet_1`, `wall_1`, `selection_1`, `motion_1` — and accepts those
identifiers in later calls.

---

## A typical session

```
rocky_status
rocky_create_project
rocky_create_surface  {kind: circular, radius: 0.3}          → surface_1
rocky_create_particle {shape: sphere, sieve_size: 0.05}      → particle_1
rocky_create_inlet    {surface_id: surface_1, particle_id: particle_1, mass_flow_rate: 1000}
rocky_set_domain      {min_values: [-5,-1.5,-1.5], max_values: [5,1.5,1.5]}
rocky_set_simulation_settings {duration: 2, time_interval: 0.1}
rocky_describe_study                                          → verify the setup
rocky_run_simulation   {non_blocking: false}
rocky_get_curve        {name: "Particles Count"}
rocky_get_curve        {name: "Particles Mass Flow In", unit: "t/h"}
rocky_save_project     {path: "C:/work/chute.rocky"}
```

---

## Configuration

Command line flags take precedence over environment variables, which take precedence over
the defaults.

| Flag | Environment variable | Default | Meaning |
|---|---|---|---|
| `--backend` | `ROCKY_MCP_BACKEND` | `auto` | `auto`, `pyrocky` or `mock`. |
| `--rocky-exe` | `ROCKY_MCP_EXE` | auto-detected | Explicit Rocky executable. |
| `--rocky-version` | `ROCKY_MCP_VERSION` | latest | Ansys version code (`261` = 2026 R1). |
| `--rocky-port` | `ROCKY_MCP_PORT` | `18615` | PyRocky RPC port. |
| `--headless` / `--no-headless` | `ROCKY_MCP_HEADLESS` | headless | GUI or headless Rocky. |
| `--no-python-session` | `ROCKY_MCP_PYTHON_SESSION` | enabled | Disable the persistent Python session. |
| `--mock-speed` | `ROCKY_MCP_MOCK_SPEED` | `8.0` | Simulated seconds per wall-clock second (mock only). |
| — | `ROCKY_MCP_WORKDIR` | current directory | Working directory. |

Transport flags come from `ansys-common-mcp`: `--transport stdio|http`,
`--http-host`, `--http-port`, `--http-path`, `--cors-origins`.

---

## Architecture

```
src/ansys_rocky_mcp/
├── server.py            RockyMCP: extends PyAnsysBaseMCP, registers the tools
├── context.py           RockyContext: session, config, persistent Python session
├── session.py           RockySession: lazy connection, object registry
├── config.py            RockyConfig: CLI / environment resolution
├── registry.py          ObjectRegistry: stable ids for Rocky API objects
├── backends/
│   ├── base.py          RockyBackend interface + BackendStatus
│   ├── pyrocky_backend.py   Real Rocky via ansys-rocky-core
│   └── mock_backend.py      In-memory Rocky simulator
└── tools/
    ├── session_tools.py     status / connect / disconnect
    ├── project_tools.py     project lifecycle + study description
    ├── setup_tools.py       geometry, particles, inlets, domain, solver, motion
    ├── run_tools.py         start / poll / wait
    └── results_tools.py     curves, grid functions, selections, Python escape hatch
```

The tool layer is written against the **Rocky application object** — the same interface
`pyrocky` returns from `RockyClient.api()` (`CreateProject`, `GetProject`, `GetVersion`,
…). Both backends expose that interface, so tool code is identical whether a real Rocky
application or the simulator is behind it.

The Rocky method names and signatures used here come from the official
[`ansys/pyrocky`](https://github.com/ansys/pyrocky) type stubs (`src/ansys/rocky/app/*.pyi`)
and the upstream `examples/basic_examples/` scripts.

---

## Development

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
.venv/Scripts/python.exe -m ruff check src tests
.venv/Scripts/python.exe -m ruff format src tests
```

The test suite covers the mock backend physics, every tool, the configuration and CLI
handling, the persistent Python session, and a **real MCP handshake over `stdio`** with
the server running as a subprocess.

---

## Limitations

- **The `pyrocky` backend has not been executed against a live Ansys Rocky installation**
  in this repository's development environment, because no Ansys software is installed
  there. It is written against the authoritative Rocky API surface, and it fails loudly
  with an actionable message when Rocky is absent. Expect to shake out version-specific
  differences on the first run against a real installation.
- The mock backend models bulk transport, not particle–particle contact mechanics. It is
  a simulator for interface exploration and testing, **not** a DEM solver.
- Result export through the Rocky export toolkit is not exposed as a tool yet.

## License

MIT — see [LICENSE](LICENSE).
