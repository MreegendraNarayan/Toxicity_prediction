"""Regenerate REST API-compatible GNN checkpoints from the project root."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from toxicity.inference.train_checkpoints import main


if __name__ == "__main__":
    main()
