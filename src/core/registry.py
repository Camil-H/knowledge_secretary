"""Tiny name->callable registries. One instance per plugin kind.

Adding a new source protocol, enricher, deliverer, or task never touches the
dispatcher: you register into one of these and the framework discovers it.
"""


class Registry:
    def __init__(self, label: str):
        self.label = label
        self._d: dict = {}

    def register(self, name: str):
        def deco(fn):
            self._d[name] = fn
            return fn

        return deco

    def get(self, name: str):
        if name not in self._d:
            raise KeyError(f"no {self.label} registered as {name!r} (have: {sorted(self._d)})")
        return self._d[name]

    def all(self) -> dict:
        return dict(self._d)


sources = Registry("source")
enrichers = Registry("enricher")
deliverers = Registry("deliverer")
tasks = Registry("task")
