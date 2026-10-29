# MCP client setup

The server speaks standard MCP over `stdio` (default) or `http`. Every client below just
needs the command, the arguments and — for the real solver — a working Ansys Rocky
installation.

Two commands are used throughout:

```bash
# Windows
PY="N:\ansys-rocky-mcp\.venv\Scripts\python.exe"

# Linux / macOS
PY="/path/to/ansys-rocky-mcp/.venv/bin/python"
```

Swap `--backend mock` for `--backend auto` once Ansys Rocky 2024 R2+ is installed and
`AWP_ROOT242` (or newer) is set.

---

## Claude Desktop

**Config file**

| OS | Path |
|---|---|
| Windows | `%APPDATA%\Claude\claude_desktop_config.json` |
| macOS | `~/Library/Application Support/Claude/claude_desktop_config.json` |
| Linux | `~/.config/Claude/claude_desktop_config.json` |

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

Restart Claude Desktop after editing. The tools appear under the connector icon.

---

## Claude Code

```bash
# User scope (available in every project)
claude mcp add --scope user ansys-rocky "N:\ansys-rocky-mcp\.venv\Scripts\python.exe" -m ansys_rocky_mcp --backend mock

# Project scope (checked into .mcp.json)
claude mcp add --scope project ansys-rocky "N:\ansys-rocky-mcp\.venv\Scripts\python.exe" -m ansys_rocky_mcp --backend mock
```

Verify with `claude mcp list`.

---

## MiniMax Code (MCode)

**Config file:** `~/.minimax/mcp.json` (`%USERPROFILE%\.minimax\mcp.json` on Windows)

```json
{
  "mcpServers": {
    "ansys-rocky": {
      "type": "stdio",
      "enabled": true,
      "command": "N:\\ansys-rocky-mcp\\.venv\\Scripts\\python.exe",
      "args": ["-m", "ansys_rocky_mcp", "--backend", "mock"],
      "timeout_ms": 300000
    }
  }
}
```

`timeout_ms` raises the per-tool timeout, which matters for long blocking runs.

MiniMax Code also manages these servers natively, so an assistant can add the entry for
you without editing the file by hand — ask it to register the server and it will persist
the same configuration.

Check the current state with:

```bash
mavis mcp list
mavis mcp get ansys-rocky
```

---

## Codex CLI

**Config file:** `~/.codex/config.toml`

```toml
[mcp_servers.ansys-rocky]
command = "N:\\ansys-rocky-mcp\\.venv\\Scripts\\python.exe"
args = ["-m", "ansys_rocky_mcp", "--backend", "mock"]
startup_timeout_sec = 90
tool_timeout_sec = 300
```

`startup_timeout_sec` matters: the server starts a persistent Python subprocess on boot.
Raise `tool_timeout_sec` if you run long simulations through the blocking
`rocky_run_simulation` call.

---

## Cursor

**Config file:** `%APPDATA%\Cursor\User\settings.json` (Windows)

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

---

## Kimi and other clients

Any MCP client that reads the standard `mcpServers` JSON block works:

```json
{
  "mcpServers": {
    "ansys-rocky": {
      "command": "/path/to/ansys-rocky-mcp/.venv/bin/python",
      "args": ["-m", "ansys_rocky_mcp", "--backend", "mock"]
    }
  }
}
```

Check your client's own documentation for where it stores that block — the format is
standard, the path is not.

---

## HTTP transport

If your client prefers a remote server, run the server once and point the client at it:

```bash
N:\ansys-rocky-mcp\.venv\Scripts\python.exe -m ansys_rocky_mcp \
    --backend mock --transport http --http-host 127.0.0.1 --http-port 8080
```

The MCP endpoint is `http://127.0.0.1:8080/mcp` by default.

```json
{
  "mcpServers": {
    "ansys-rocky": {
      "type": "http",
      "url": "http://127.0.0.1:8080/mcp"
    }
  }
}
```

---

## Verifying the connection

Ask the client to call the `rocky_status` tool. A healthy response looks like this:

```json
{
  "backend": "mock",
  "simulated": true,
  "connected": true,
  "rocky_available": false,
  "rocky_version": "2026R1 (mock)",
  "installed_versions": [],
  "detail": "Running against the built-in Rocky simulator. Results are physically consistent but SIMULATED, not computed by Ansys Rocky."
}
```

With a real Ansys installation the same call reports `"backend": "pyrocky"`,
`"simulated": false` and the actual Rocky version.

---

## Troubleshooting

**`No Ansys Rocky application is connected yet`**
The backend has not been connected. Call `rocky_connect` first, or use a backend that
connects automatically.

**`Ansys Rocky was not detected on this machine`**
The `auto` backend found no local Ansys installation. Install Ansys Rocky 2024 R2+ and
make sure `AWP_ROOT242` (or newer) is set, or switch to `--backend mock` to explore the
interface.

**The server starts but no tools appear**
Check that the command path points at the virtual environment's interpreter and that
`ansys-rocky-mcp` is installed there:

```bash
N:\ansys-rocky-mcp\.venv\Scripts\python.exe -m pip show ansys-rocky-mcp
```

**Timeouts during long simulations**
`rocky_run_simulation` with `non_blocking: false` blocks for the whole run. Prefer
`non_blocking: true` plus `rocky_simulation_status` / `rocky_wait_for_completion`, and
raise the client's tool timeout.
