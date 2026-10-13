from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


SUPERCOMPUTERS_DIRNAME = "supercomputers"
LEGACY_SUPERCOMPUTERS_FILENAME = "supercomputers.json"
NODE_CONFIG_FILENAME = "node_config.json"
JOBS_DIRNAME = "jobs"


def sanitize_identifier(value: str) -> str:
    identifier = re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip().lower())
    identifier = re.sub(r"_+", "_", identifier).strip("_")
    if not identifier:
        raise ValueError("Identifier cannot be empty.")
    return identifier


def normalize_supercomputer_config(system_id: str, config: dict[str, Any]) -> dict[str, Any]:
    if "clusters" in config:
        normalized = dict(config)
    else:
        normalized = {"clusters": dict(config)}

    clusters = normalized.get("clusters")
    if not isinstance(clusters, dict) or not clusters:
        raise ValueError(f"Supercomputer '{system_id}' must define at least one cluster.")

    normalized["id"] = normalized.get("id", system_id)
    normalized["clusters"] = {
        str(cluster_type): int(num_nodes)
        for cluster_type, num_nodes in clusters.items()
    }
    return normalized


def load_supercomputers_config(datasets_dir: str | Path) -> dict[str, dict[str, Any]]:
    input_path = Path(datasets_dir)
    if input_path.is_file():
        with input_path.open("r", encoding="utf-8") as f:
            legacy = json.load(f)
        return {
            sanitize_identifier(system_id): normalize_supercomputer_config(system_id, config)
            for system_id, config in legacy.items()
        }

    datasets_path = input_path
    supercomputers_dir = datasets_path / SUPERCOMPUTERS_DIRNAME
    systems: dict[str, dict[str, Any]] = {}

    if supercomputers_dir.is_dir():
        for path in sorted(supercomputers_dir.glob("*.json")):
            with path.open("r", encoding="utf-8") as f:
                raw_config = json.load(f)
            system_id = sanitize_identifier(raw_config.get("id") or path.stem)
            systems[system_id] = normalize_supercomputer_config(system_id, raw_config)

    if systems:
        return systems

    legacy_path = datasets_path / LEGACY_SUPERCOMPUTERS_FILENAME
    if legacy_path.exists():
        with legacy_path.open("r", encoding="utf-8") as f:
            legacy = json.load(f)
        return {
            sanitize_identifier(system_id): normalize_supercomputer_config(system_id, config)
            for system_id, config in legacy.items()
        }

    return {}


def save_supercomputer_config(
    datasets_dir: str | Path,
    system_id: str,
    config: dict[str, Any],
) -> Path:
    datasets_path = Path(datasets_dir)
    supercomputers_dir = datasets_path / SUPERCOMPUTERS_DIRNAME
    supercomputers_dir.mkdir(parents=True, exist_ok=True)

    system_id = sanitize_identifier(system_id)
    normalized = normalize_supercomputer_config(system_id, {**config, "id": system_id})
    out_path = supercomputers_dir / f"{system_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=4)
        f.write("\n")
    return out_path


def load_node_config(datasets_dir: str | Path) -> dict[str, dict[str, Any]]:
    input_path = Path(datasets_dir)
    path = input_path if input_path.is_file() else input_path / NODE_CONFIG_FILENAME
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def jobs_dataset_filename(cluster_type: str, queue_name: str) -> str:
    return f"jobs_{sanitize_identifier(cluster_type)}_{sanitize_identifier(queue_name)}.csv"


def jobs_dataset_dir(datasets_dir: str | Path, system_id: str) -> Path:
    return Path(datasets_dir) / JOBS_DIRNAME / sanitize_identifier(system_id)


def jobs_dataset_path(
    datasets_dir: str | Path,
    system_id: str,
    cluster_type: str,
    queue_name: str,
) -> Path:
    return jobs_dataset_dir(datasets_dir, system_id) / jobs_dataset_filename(
        cluster_type,
        queue_name,
    )


def save_node_config(datasets_dir: str | Path, node_config: dict[str, Any]) -> Path:
    path = Path(datasets_dir) / NODE_CONFIG_FILENAME
    with path.open("w", encoding="utf-8") as f:
        json.dump(node_config, f, indent=4)
        f.write("\n")
    return path
