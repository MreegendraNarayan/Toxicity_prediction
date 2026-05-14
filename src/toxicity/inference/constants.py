"""Shared inference constants."""

from typing import Dict

TOX21_TASKS = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

DEFAULT_MODEL_NAME = "Enhanced-GIN"

MODEL_SPECS: Dict[str, str] = {
    "Enhanced-GCN": "gcn",
    "Enhanced-GIN": "gin",
}

FEATURE_VERSION = "rdkit-topology-v1"
