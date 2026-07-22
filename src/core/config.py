"""Load config.yaml. Kept deliberately thin — the schema lives in the file itself
and in CONTRACTS.md."""

import yaml


def load(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
