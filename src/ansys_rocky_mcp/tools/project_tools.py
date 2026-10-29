"""Project-level tools for the Rocky MCP server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import Context
from fastmcp.exceptions import ToolError

from ansys_rocky_mcp.tools._common import envelope, get_session

__all__ = ["register"]


def _class_name(element: Any) -> str:
    """Return the Rocky class name of an element, or an empty string."""
    getter = getattr(element, "GetClassName", None)
    if getter is None:
        return ""
    try:
        return str(getter())
    except Exception:  # pragma: no cover - remote calls may fail
        return ""


def _safe(call: Any, default: Any = None) -> Any:
    """Call ``call`` and return ``default`` when it fails."""
    try:
        return call()
    except Exception:
        return default


def _describe_element(element: Any) -> dict[str, Any]:
    """Describe a single study element without raising."""
    description: dict[str, Any] = {
        "class": _class_name(element),
        "name": _safe(getattr(element, "GetName", lambda: None)),
    }
    if hasattr(element, "GetMaxRadius"):
        description["max_radius_m"] = _safe(lambda: element.GetMaxRadius("m"))
    if hasattr(element, "GetLength"):
        description["length_m"] = _safe(lambda: element.GetLength("m"))
        description["width_m"] = _safe(lambda: element.GetWidth("m"))
    if hasattr(element, "GetCenter"):
        description["center_m"] = _safe(lambda: list(element.GetCenter("m")))
    if hasattr(element, "GetShape"):
        description["shape"] = _safe(element.GetShape)
    if hasattr(element, "GetMinimumSize"):
        description["sieve_size_m"] = _safe(lambda: element.GetMinimumSize("m"))
    if hasattr(element, "GetInputPropertiesList"):
        properties = _safe(element.GetInputPropertiesList, []) or []
        description["mass_flow_t_h"] = _safe(
            lambda: properties[0].GetMassFlowRate("t/h") if properties else None
        )
    return description


def describe_study(session: Any) -> dict[str, Any]:
    """Return a JSON-serializable snapshot of the open project and study."""
    project = session.get_project()
    study = project.GetStudy()

    element_names = _safe(lambda: study.GetElementNames(), []) or []
    elements: list[dict[str, Any]] = []
    for name in element_names:
        element = _safe(lambda name=name: study.GetElement(name))
        if element is None:
            elements.append({"name": name, "class": ""})
            continue
        elements.append(_describe_element(element))

    domain = _safe(study.GetDomainSettings)
    solver = _safe(study.GetSolver)
    processes = _safe(study.GetUserProcessCollection)
    process_names = _safe(lambda: list(processes.GetProcessNames()), []) or [] if processes else []

    summary: dict[str, Any] = {}
    if hasattr(study, "to_dict"):
        summary = study.to_dict()

    return {
        "project": {
            "filename": _safe(project.GetProjectFilename, "") or "",
            "has_unsaved_changes": _safe(project.HasUnsavedChanges, False),
            "studies": _safe(project.GetStudyNames, []) or [],
        },
        "study": {
            "name": _safe(getattr(study, "GetName", lambda: "Study 1")),
            "elements": elements,
            "domain": {
                "min_m": _safe(lambda: list(domain.GetCoordinateLimitsMinValues("m")), []),
                "max_m": _safe(lambda: list(domain.GetCoordinateLimitsMaxValues("m")), []),
                "use_boundary_limits": _safe(domain.GetUseBoundaryLimits),
            }
            if domain is not None
            else {},
            "solver": {
                "duration_s": _safe(lambda: solver.GetSimulationDuration("s")),
                "time_interval_s": _safe(lambda: solver.GetTimeInterval("s")),
                "processors": _safe(solver.GetNumberOfProcessors),
                "execution_mode": _safe(solver.GetExecutionMode),
            }
            if solver is not None
            else {},
            "processes": process_names,
            "simulation": {
                "simulating": _safe(study.IsSimulating, False),
                "progress_percent": _safe(study.GetProgress),
                "has_results": _safe(study.HasResults, False),
            },
        },
        "summary": summary,
        "registered_objects": session.registry.list(),
    }


async def rocky_create_project(ctx: Context) -> dict[str, Any]:
    """Create a new, empty Rocky project.

    Any previously open project is replaced and the object registry is reset.
    """
    session = get_session(ctx)
    app = session.ensure_app()
    project = app.CreateProject()
    session.registry.clear()
    return envelope(
        session,
        message="Rocky project created.",
        project_filename=_safe(project.GetProjectFilename, "") or "",
    )


async def rocky_open_project(ctx: Context, path: str) -> dict[str, Any]:
    """Open an existing Rocky project file.

    Parameters
    ----------
    path : str
        Path to the ``.rocky`` project file.

    """
    session = get_session(ctx)
    app = session.ensure_app()
    target = Path(path).expanduser()
    if not target.exists():
        raise ToolError(f"Rocky project file not found: {target}")
    try:
        project = app.OpenProject(str(target))
    except Exception as exc:
        raise ToolError(f"Could not open the Rocky project {target}: {exc}") from exc
    session.registry.clear()
    return envelope(
        session,
        message=f"Opened Rocky project {target}.",
        project_filename=_safe(project.GetProjectFilename, str(target)),
    )


async def rocky_save_project(ctx: Context, path: str | None = None) -> dict[str, Any]:
    """Save the open Rocky project.

    Parameters
    ----------
    path : str | None, default: None
        Destination file. When omitted, the project is saved to its current
        path.

    """
    session = get_session(ctx)
    project = session.get_project()
    try:
        if path:
            project.SaveProject(str(Path(path).expanduser()))
        else:
            project.SaveProject()
    except Exception as exc:
        raise ToolError(f"Could not save the Rocky project: {exc}") from exc
    return envelope(
        session,
        message="Rocky project saved.",
        project_filename=_safe(project.GetProjectFilename, "") or "",
    )


async def rocky_close_project(ctx: Context) -> dict[str, Any]:
    """Close the open Rocky project."""
    session = get_session(ctx)
    app = session.ensure_app()
    try:
        app.CloseProject()
    except Exception as exc:
        raise ToolError(f"Could not close the Rocky project: {exc}") from exc
    session.registry.clear()
    return envelope(session, message="Rocky project closed.")


async def rocky_describe_study(ctx: Context) -> dict[str, Any]:
    """Describe the open project and study.

    Returns the project path, every study element with its class and key
    parameters, the domain and solver settings, the user processes, the
    simulation state, and the identifiers of the registered objects.

    Call this after any setup step to confirm what the study actually
    contains.
    """
    session = get_session(ctx)
    try:
        description = describe_study(session)
    except RuntimeError as exc:
        raise ToolError(str(exc)) from exc
    return envelope(session, **description)


def register(mcp: Any) -> None:
    """Register the project tools on the MCP server."""
    mcp.tool()(rocky_create_project)
    mcp.tool()(rocky_open_project)
    mcp.tool()(rocky_save_project)
    mcp.tool()(rocky_close_project)
    mcp.tool()(rocky_describe_study)
