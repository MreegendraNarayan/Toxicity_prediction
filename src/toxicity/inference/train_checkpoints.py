"""Train compatible GCN/GIN checkpoints for the REST API.

This command regenerates checkpoints that match the API inference models:

    python train_api_checkpoints.py --models all
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch_geometric.loader import DataLoader

from toxicity.inference.constants import FEATURE_VERSION, MODEL_SPECS, TOX21_TASKS
from toxicity.inference.graph import InvalidSmilesError, smiles_to_graph
from toxicity.inference.models import build_model
from toxicity.utils import set_seed

logger = logging.getLogger(__name__)

DEFAULT_TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"


def load_tox21_graphs(data_url: str, limit: int = 0) -> List:
    """Load Tox21 molecules from CSV and convert them to PyG graphs."""
    df = pd.read_csv(data_url).dropna(subset=["smiles"]).reset_index(drop=True)
    if limit:
        df = df.head(limit)

    graphs = []
    for _, row in df.iterrows():
        labels = [
            row[task] if task in row and not pd.isna(row[task]) else -1.0
            for task in TOX21_TASKS
        ]
        try:
            graphs.append(smiles_to_graph(row["smiles"], labels=labels))
        except InvalidSmilesError:
            continue

    if not graphs:
        raise RuntimeError("No valid molecular graphs were generated from Tox21")

    return graphs


def split_graphs(graphs: List, seed: int) -> Tuple[List, List, List]:
    """Create deterministic train/validation/test splits."""
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(graphs), generator=generator).tolist()
    train_end = int(0.8 * len(graphs))
    val_end = int(0.9 * len(graphs))
    return (
        [graphs[i] for i in permutation[:train_end]],
        [graphs[i] for i in permutation[train_end:val_end]],
        [graphs[i] for i in permutation[val_end:]],
    )


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        target = batch.y.to(torch.float32).view(logits.shape)
        mask = target != -1
        if mask.sum() == 0:
            continue
        loss = criterion(logits[mask], target[mask])
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(batches, 1)


def collect_predictions(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probabilities = torch.sigmoid(logits)
            target = batch.y.to(torch.float32).view(logits.shape)
            y_true.append(target.cpu().numpy())
            y_pred.append(probabilities.cpu().numpy())
    return np.vstack(y_true), np.vstack(y_pred)


def mean_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for task_idx in range(y_true.shape[1]):
        mask = y_true[:, task_idx] != -1
        if mask.sum() == 0:
            continue
        labels = y_true[mask, task_idx]
        if len(np.unique(labels)) < 2:
            continue
        scores.append(roc_auc_score(labels, y_pred[mask, task_idx]))
    return float(np.mean(scores)) if scores else 0.5


def train_model(
    model_name: str,
    architecture: str,
    train_loader,
    val_loader,
    test_loader,
    checkpoint_dir: Path,
    epochs: int,
    learning_rate: float,
    device,
) -> Dict[str, object]:
    model = build_model(architecture, num_tasks=len(TOX21_TASKS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        y_val, p_val = collect_predictions(model, val_loader, device)
        val_auc = mean_roc_auc(y_val, p_val)
        history.append({"epoch": epoch, "train_loss": loss, "val_roc_auc": val_auc})
        logger.info("%s epoch %s/%s loss=%.4f val_auc=%.4f", model_name, epoch, epochs, loss, val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {key: value.cpu() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test, p_test = collect_predictions(model, test_loader, device)
    test_auc = mean_roc_auc(y_test, p_test)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_best.pt"
    torch.save(
        {
            "model_name": model_name,
            "architecture": architecture,
            "task_names": TOX21_TASKS,
            "feature_version": FEATURE_VERSION,
            "created_at": datetime.now().isoformat(),
            "validation_roc_auc": best_val_auc,
            "test_roc_auc": test_auc,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    return {
        "model_name": model_name,
        "architecture": architecture,
        "checkpoint_path": str(checkpoint_path),
        "validation_roc_auc": best_val_auc,
        "test_roc_auc": test_auc,
        "history": history,
    }


def parse_models(value: str) -> Iterable[str]:
    if value == "all":
        return MODEL_SPECS.keys()
    requested = [item.strip() for item in value.split(",") if item.strip()]
    for item in requested:
        if item not in MODEL_SPECS:
            raise argparse.ArgumentTypeError(f"Unsupported model: {item}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Train REST API-compatible GNN checkpoints")
    parser.add_argument("--models", default="all", help="all, Enhanced-GCN, Enhanced-GIN, or comma-separated names")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--checkpoint-dir", default="models/checkpoints")
    parser.add_argument("--data-url", default=DEFAULT_TOX21_URL)
    parser.add_argument("--limit", type=int, default=0, help="Optional molecule limit for quick smoke training")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graphs = load_tox21_graphs(args.data_url, limit=args.limit)
    train_graphs, val_graphs, test_graphs = split_graphs(graphs, args.seed)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)

    checkpoint_dir = Path(args.checkpoint_dir)
    results = []
    for model_name in parse_models(args.models):
        results.append(
            train_model(
                model_name=model_name,
                architecture=MODEL_SPECS[model_name],
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                checkpoint_dir=checkpoint_dir,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                device=device,
            )
        )

    results_path = Path("results") / "api_training_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with results_path.open("w", encoding="utf-8") as file:
        json.dump({"created_at": datetime.now().isoformat(), "models": results}, file, indent=2)

    logger.info("Training complete. Results written to %s", results_path)


if __name__ == "__main__":
    main()
