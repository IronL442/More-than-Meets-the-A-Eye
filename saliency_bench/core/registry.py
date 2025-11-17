"""
Simple string-to-class registry for models and datasets.
"""

_REGISTRY = {"model": {}, "dataset": {}}


def register(kind, name):
    if kind not in _REGISTRY:
        raise KeyError(f"Unsupported registry kind '{kind}'")

    def deco(cls):
        _REGISTRY[kind][name] = cls
        return cls

    return deco


def build(kind, name, **kwargs):
    if name not in _REGISTRY.get(kind, {}):
        raise KeyError(
            f"{kind} '{name}' not found. Registered: {list(_REGISTRY.get(kind, {}))}"
        )
    return _REGISTRY[kind][name](**kwargs)


def list_registered(kind):
    if kind not in _REGISTRY:
        raise KeyError(f"Unsupported registry kind '{kind}'")
    return sorted(_REGISTRY[kind].keys())

