"""Load config.yaml. Kept deliberately thin — the schema lives in the file itself
and in CONTRACTS.md."""

import os

import yaml


def load(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    _expand_env(cfg)
    return cfg


def _expand_env(node):
    """In-place: replace any string of the form ${VAR} with os.environ["VAR"] (or "")."""
    if isinstance(node, dict):
        for k, v in node.items():
            node[k] = _expand_env(v)
        return node
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str) and node.startswith("${") and node.endswith("}"):
        return os.environ.get(node[2:-1], "")
    return node
