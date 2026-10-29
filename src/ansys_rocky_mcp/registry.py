"""Stable identifier registry for Ansys Rocky API objects.

The Rocky API hands back opaque proxy objects that an AI client cannot
reference later ("the inlet you just created"). This registry assigns every
object a short, stable identifier such as ``surface_1`` so that later tool
calls can address it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["ObjectRegistry", "RegisteredObject"]

#: Object categories tracked by the registry.
CATEGORIES = (
    "particle",
    "surface",
    "inlet",
    "outlet",
    "wall",
    "selection",
    "motion",
)


@dataclass
class RegisteredObject:
    """A single registered Rocky API object."""

    id: str
    category: str
    obj: Any
    label: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the registration."""
        return {
            "id": self.id,
            "category": self.category,
            "label": self.label,
            **({"meta": self.meta} if self.meta else {}),
        }


class ObjectRegistry:
    """Map short identifiers to live Rocky API objects.

    Examples
    --------
    >>> registry = ObjectRegistry()
    >>> ident = registry.add("surface", object(), label="inlet disk")
    >>> ident
    'surface_1'
    >>> registry.get(ident) is not None
    True

    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._objects: dict[str, RegisteredObject] = {}
        self._counters: dict[str, int] = {}

    def add(
        self,
        category: str,
        obj: Any,
        label: str = "",
        meta: dict[str, Any] | None = None,
    ) -> str:
        """Register an object and return its new identifier.

        Parameters
        ----------
        category : str
            One of :data:`CATEGORIES`.
        obj : Any
            The live Rocky API object.
        label : str, default: ""
            Human-readable description shown to the AI client.
        meta : dict[str, Any] | None, default: None
            Extra JSON-serializable details (for example the surface kind).

        """
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category {category!r}. Expected one of {CATEGORIES}.")

        index = self._counters.get(category, 0) + 1
        self._counters[category] = index
        ident = f"{category}_{index}"
        self._objects[ident] = RegisteredObject(
            id=ident,
            category=category,
            obj=obj,
            label=label,
            meta=dict(meta or {}),
        )
        return ident

    def get(self, ident: str) -> RegisteredObject | None:
        """Return the registration for ``ident``, or ``None`` when unknown."""
        return self._objects.get(ident)

    def resolve(self, ident: str, category: str | None = None) -> Any:
        """Return the live object for ``ident``.

        Raises
        ------
        KeyError
            When the identifier is unknown or belongs to another category.

        """
        entry = self._objects.get(ident)
        if entry is None:
            known = ", ".join(sorted(self._objects)) or "(none)"
            raise KeyError(f"Unknown object id {ident!r}. Known ids: {known}.")
        if category is not None and entry.category != category:
            raise KeyError(f"Object {ident!r} is a {entry.category!r}, expected a {category!r}.")
        return entry.obj

    def list(self, category: str | None = None) -> list[dict[str, Any]]:
        """Return JSON-serializable views of the registered objects."""
        return [
            entry.to_dict()
            for entry in self._objects.values()
            if category is None or entry.category == category
        ]

    def clear(self) -> None:
        """Forget every registered object."""
        self._objects.clear()
        self._counters.clear()

    def __len__(self) -> int:
        """Return the number of registered objects."""
        return len(self._objects)
