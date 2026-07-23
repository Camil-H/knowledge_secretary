"""Tiny name->callable registries — one instance per plugin kind."""

from collections.abc import Callable


class Registry[T]:
    """Name->value registry for one plugin kind, populated via `@register`."""

    def __init__(self, label: str):
        self.label = label
        self._d: dict[str, T] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def deco(fn: T) -> T:
            self._d[name] = fn
            return fn

        return deco

    def get(self, name: str) -> T:
        if name not in self._d:
            raise KeyError(f"no {self.label} registered as {name!r} (have: {sorted(self._d)})")
        return self._d[name]

    def all(self) -> dict[str, T]:
        return dict(self._d)


sources = Registry("source")
enrichers = Registry("enricher")
deliverers = Registry("deliverer")
tasks = Registry("task")
