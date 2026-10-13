from __future__ import annotations

import json
import sys
from pathlib import Path

DATASETS_DIR = Path(__file__).resolve().parents[1] / "datasets"
if str(DATASETS_DIR) not in sys.path:
    sys.path.insert(0, str(DATASETS_DIR))

from supercomputer_config import (
    load_supercomputers_config,
    sanitize_identifier,
    save_supercomputer_config,
)


def test_load_supercomputers_from_directory(tmp_path):
    sc_dir = tmp_path / "supercomputers"
    sc_dir.mkdir()
    (sc_dir / "custom_hpc.json").write_text(
        json.dumps({
            "id": "custom_hpc",
            "label": "Custom HPC",
            "clusters": {"cpu_node": 10},
            "sla_policy": "enforce",
        }),
        encoding="utf-8",
    )

    systems = load_supercomputers_config(tmp_path)

    assert list(systems) == ["custom_hpc"]
    assert systems["custom_hpc"]["clusters"] == {"cpu_node": 10}
    assert systems["custom_hpc"]["label"] == "Custom HPC"


def test_save_supercomputer_writes_one_json_file(tmp_path):
    path = save_supercomputer_config(
        tmp_path,
        "My HPC",
        {"clusters": {"cpu_node": 12}},
    )

    assert path.name == "my_hpc.json"
    systems = load_supercomputers_config(tmp_path)
    assert systems["my_hpc"]["clusters"] == {"cpu_node": 12}


def test_sanitize_identifier_normalizes_user_names():
    assert sanitize_identifier(" My HPC 2026! ") == "my_hpc_2026"
