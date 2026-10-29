"""The Ansys Rocky MCP server."""

from __future__ import annotations

import argparse
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from ansys.common.mcp.helpers import PersistentPythonSession
from ansys.common.mcp.server import PyAnsysBaseMCP
from fastmcp import FastMCP

from ansys_rocky_mcp.config import BACKEND_AUTO, BACKEND_MOCK, BACKEND_PYROCKY, RockyConfig
from ansys_rocky_mcp.context import RockyContext
from ansys_rocky_mcp.session import RockySession
from ansys_rocky_mcp.tools import register_all

__all__ = ["INSTRUCTIONS", "ROCKY_STARTUP_CODE", "RockyMCP"]

#: Code executed once when the persistent Python session starts.
ROCKY_STARTUP_CODE = """
import sys

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - depends on the environment
    print(f"numpy unavailable: {exc}")

try:
    import ansys.rocky.core as pyrocky

    print(f"PyRocky available: {pyrocky.__name__}")
except Exception as exc:  # pragma: no cover - depends on the environment
    print(f"PyRocky unavailable: {exc}")
"""

INSTRUCTIONS = """
Drive Ansys Rocky, the discrete element method (DEM) solver, from an AI client.

Typical workflow:

1. `rocky_status` - check the backend and whether Ansys Rocky is available.
2. `rocky_connect` - launch or attach to Rocky (only needed with the real solver).
3. `rocky_create_project` (or `rocky_open_project`).
4. Build the study: `rocky_import_wall`, `rocky_create_surface`,
   `rocky_create_particle`, `rocky_create_inlet`, `rocky_create_outlet`,
   `rocky_set_domain`, `rocky_set_simulation_settings`, `rocky_set_motion`.
5. `rocky_describe_study` - verify the setup before running.
6. `rocky_run_simulation` then `rocky_simulation_status` /
   `rocky_wait_for_completion`.
7. `rocky_get_curve`, `rocky_get_grid_function_stats`, `rocky_create_selection`
   for results, and `rocky_save_project` to persist the project.

Objects created by the tools are addressed by short identifiers such as
`surface_1` or `particle_1`; pass those identifiers to the later tools.

Every response carries `backend` and `simulated`. When `simulated` is true the
numbers come from the built-in simulator and are NOT Ansys Rocky results -
say so explicitly rather than reporting them as solver output.

Use `rocky_run_python` as an escape hatch for anything the dedicated tools do
not cover.
"""


class RockyMCP(PyAnsysBaseMCP):
    """MCP server exposing Ansys Rocky through the PyAnsys Common MCP stack.

    The server follows the same extension pattern as the other PyAnsys MCP
    servers: it extends :class:`PyAnsysBaseMCP`, implements
    :meth:`product_startup` and :meth:`product_cleanup`, and exposes the
    product calls as MCP tools.
    """

    def __init__(
        self,
        *args: Any,
        config: RockyConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the server.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to FastMCP.
        config : RockyConfig | None, default: None
            Server configuration. Defaults to the environment configuration.
        **kwargs : Any
            Keyword arguments passed to FastMCP.

        """
        self.config = config or RockyConfig.from_env()
        super().__init__(*args, need_python=True, **kwargs)
        self.register_tools()

    # ------------------------------------------------------------------
    # PyAnsysBaseMCP contract
    # ------------------------------------------------------------------
    def create_context(self) -> RockyContext:
        """Create the Rocky application context."""
        session = RockySession(self.config)
        python_session = None
        if self.config.python_session:
            python_session = PersistentPythonSession(
                python_executable=self.python_executable,
                startup_code=ROCKY_STARTUP_CODE,
                working_directory=self.working_directory,
            )
        return RockyContext(
            product_instance=session,
            python_executable=self.python_executable,
            python_session=python_session,
            session=session,
            config=self.config,
            command_history=[],
        )

    def product_startup(self) -> None:
        """Report the backend state when the server starts.

        Rocky is deliberately *not* launched here: launching Ansys takes
        tens of seconds and the first tool call should decide when that
        happens.
        """
        context = getattr(self, "context", None)
        session = getattr(context, "session", None)
        if session is None:
            return
        status = session.status()
        if status.detail:
            print(f"[ansys-rocky-mcp] {status.detail}", file=sys.stderr)

    def product_cleanup(self) -> None:
        """Close the Rocky session when the server shuts down."""
        context = getattr(self, "context", None)
        session = getattr(context, "session", None)
        if session is not None:
            session.close()

    @asynccontextmanager
    async def product_lifespan(self, server: FastMCP) -> AsyncIterator[RockyContext]:
        """Manage the lifetime of the Rocky context."""
        self.server = server
        self.context = self.create_context()
        self.product_startup()
        if self.config.python_session and self.context.python_session is not None:
            self.start_python_session()
        try:
            yield self.context
        finally:
            if self.config.python_session and self.context.python_session is not None:
                self.cleanup_python_session()
            self.product_cleanup()

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    def _add_cli_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Add the Rocky-specific command line arguments."""
        parser.add_argument(
            "--backend",
            dest="backend",
            choices=[BACKEND_AUTO, BACKEND_PYROCKY, BACKEND_MOCK],
            default=None,
            help=(
                "Backend that drives the tools. 'auto' uses Ansys Rocky when it is "
                "installed and fails with an actionable error otherwise; 'mock' uses "
                "the built-in simulator and tags every result as simulated."
            ),
        )
        parser.add_argument(
            "--rocky-exe",
            dest="rocky_exe",
            default=None,
            help="Explicit path to the Rocky executable.",
        )
        parser.add_argument(
            "--rocky-version",
            dest="rocky_version",
            type=int,
            default=None,
            help="Ansys version code of Rocky to launch, for example 261 for 2026 R1.",
        )
        parser.add_argument(
            "--rocky-port",
            dest="rocky_port",
            type=int,
            default=None,
            help="Port for the PyRocky RPC server.",
        )
        parser.add_argument(
            "--headless",
            dest="headless",
            action="store_true",
            default=None,
            help="Launch Rocky without a graphical user interface.",
        )
        parser.add_argument(
            "--no-headless",
            dest="headless",
            action="store_false",
            help="Launch Rocky with a graphical user interface.",
        )
        parser.add_argument(
            "--no-python-session",
            dest="python_session",
            action="store_false",
            default=None,
            help="Disable the persistent Python session used by 'rocky_run_python'.",
        )
        parser.add_argument(
            "--mock-speed",
            dest="mock_speed",
            type=float,
            default=None,
            help="Simulated seconds per wall-clock second used by the mock backend.",
        )

    def _configure_from_cli(self, args: argparse.Namespace) -> None:
        """Apply the parsed command line arguments to the configuration."""
        self.config = self.config.merge(
            backend=getattr(args, "backend", None),
            rocky_exe=getattr(args, "rocky_exe", None),
            rocky_version=getattr(args, "rocky_version", None),
            rocky_port=getattr(args, "rocky_port", None),
            headless=getattr(args, "headless", None),
            python_session=getattr(args, "python_session", None),
            mock_speed=getattr(args, "mock_speed", None),
        )
        self._cli_config = {"backend": self.config.backend}

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------
    def register_tools(self) -> None:
        """Register every Rocky tool on the server."""
        register_all(self)
