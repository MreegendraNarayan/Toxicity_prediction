"""High-level cached model loading and prediction service."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch_geometric.data import Batch

from .constants import DEFAULT_MODEL_NAME, FEATURE_VERSION, MODEL_SPECS, TOX21_TASKS
from .graph import InvalidSmilesError, smiles_to_graph
from .models import build_model, predict_probabilities

logger = logging.getLogger(__name__)


class ModelUnavailableError(RuntimeError):
    """Raised when a requested model checkpoint is missing or incompatible."""


@dataclass
class ModelStatus:
    name: str
    architecture: str
    checkpoint_path: str
    exists: bool
    compatible: bool
    size_kb: Optional[float]
    message: str


def risk_label(average_probability: float) -> str:
    """Map average toxicity probability to a simple risk label."""
    if average_probability < 0.33:
        return "low"
    if average_probability < 0.66:
        return "moderate"
    return "high"


class ToxicityPredictor:
    """Cached checkpoint loader and inference facade."""

    def __init__(
        self,
        checkpoint_dir: str = "models/checkpoints",
        device: str = "cpu",
        default_model: str = DEFAULT_MODEL_NAME,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = torch.device(device)
        self.default_model = default_model
        self._cache: Dict[str, torch.nn.Module] = {}

    def checkpoint_path(self, model_name: str) -> Path:
        return self.checkpoint_dir / f"{model_name}_best.pt"

    def available_models(self) -> List[ModelStatus]:
        """Return status for each supported API model."""
        return [self.check_model_status(name) for name in MODEL_SPECS]

    def check_model_status(self, model_name: str) -> ModelStatus:
        """Inspect whether a model checkpoint is present and load-compatible."""
        architecture = self._architecture_for(model_name)
        path = self.checkpoint_path(model_name)
        if not path.exists():
            return ModelStatus(
                name=model_name,
                architecture=architecture,
                checkpoint_path=str(path),
                exists=False,
                compatible=False,
                size_kb=None,
                message="Checkpoint is missing. Run `python train_api_checkpoints.py`.",
            )

        try:
            state_dict, metadata = self._load_checkpoint_state(path)
            model = build_model(architecture, num_tasks=len(TOX21_TASKS))
            model.load_state_dict(state_dict, strict=True)

            metadata_tasks = metadata.get("task_names")
            if metadata_tasks is not None and list(metadata_tasks) != TOX21_TASKS:
                raise ModelUnavailableError("Checkpoint task names do not match the 12 Tox21 API tasks")

            return ModelStatus(
                name=model_name,
                architecture=architecture,
                checkpoint_path=str(path),
                exists=True,
                compatible=True,
                size_kb=path.stat().st_size / 1024,
                message="Compatible",
            )
        except Exception as exc:
            return ModelStatus(
                name=model_name,
                architecture=architecture,
                checkpoint_path=str(path),
                exists=True,
                compatible=False,
                size_kb=path.stat().st_size / 1024,
                message=(
                    f"Incompatible checkpoint: {exc}. "
                    "Run `python train_api_checkpoints.py` to regenerate it."
                ),
            )

    def predict_one(self, smiles: str, model_name: Optional[str] = None) -> Dict[str, object]:
        """Predict toxicity probabilities for one molecule."""
        model_key = model_name or self.default_model
        model = self._load_model(model_key)
        graph = smiles_to_graph(smiles)
        batch = Batch.from_data_list([graph]).to(self.device)
        probabilities = predict_probabilities(model, batch)[0].cpu().tolist()
        return self._format_prediction(graph.smiles, model_key, probabilities)

    def predict_many(self, smiles_values: Iterable[str], model_name: Optional[str] = None) -> List[Dict[str, object]]:
        """Predict toxicity probabilities for multiple molecules."""
        model_key = model_name or self.default_model
        model = self._load_model(model_key)
        graphs = [smiles_to_graph(smiles) for smiles in smiles_values]
        batch = Batch.from_data_list(graphs).to(self.device)
        probabilities = predict_probabilities(model, batch).cpu().tolist()
        return [
            self._format_prediction(graph.smiles, model_key, row)
            for graph, row in zip(graphs, probabilities)
        ]

    def _load_model(self, model_name: str) -> torch.nn.Module:
        if model_name in self._cache:
            return self._cache[model_name]

        architecture = self._architecture_for(model_name)
        path = self.checkpoint_path(model_name)
        if not path.exists():
            raise ModelUnavailableError(
                f"Checkpoint not found for {model_name}: {path}. "
                "Run `python train_api_checkpoints.py`."
            )

        try:
            state_dict, metadata = self._load_checkpoint_state(path)
            metadata_tasks = metadata.get("task_names")
            if metadata_tasks is not None and list(metadata_tasks) != TOX21_TASKS:
                raise ModelUnavailableError("Checkpoint task names do not match the 12 Tox21 API tasks")

            model = build_model(architecture, num_tasks=len(TOX21_TASKS))
            model.load_state_dict(state_dict, strict=True)
            model.to(self.device)
            model.eval()
            self._cache[model_name] = model
            logger.info("Loaded %s checkpoint from %s", model_name, path)
            return model
        except ModelUnavailableError:
            raise
        except Exception as exc:
            raise ModelUnavailableError(
                f"Checkpoint for {model_name} is incompatible: {exc}. "
                "Run `python train_api_checkpoints.py`."
            ) from exc

    def _architecture_for(self, model_name: str) -> str:
        if model_name not in MODEL_SPECS:
            supported = ", ".join(MODEL_SPECS)
            raise ModelUnavailableError(f"Unsupported model '{model_name}'. Supported models: {supported}")
        return MODEL_SPECS[model_name]

    @staticmethod
    def _load_checkpoint_state(path: Path) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
        checkpoint = torch.load(path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"], dict(checkpoint)
        if isinstance(checkpoint, dict):
            return checkpoint, {}
        raise ModelUnavailableError("Unsupported checkpoint format")

    @staticmethod
    def _format_prediction(smiles: str, model_name: str, probabilities: List[float]) -> Dict[str, object]:
        predictions = {
            task: float(probability)
            for task, probability in zip(TOX21_TASKS, probabilities)
        }
        average = float(sum(predictions.values()) / len(predictions))
        return {
            "model": model_name,
            "smiles": smiles,
            "predictions": predictions,
            "average_toxicity": average,
            "risk_label": risk_label(average),
            "feature_version": FEATURE_VERSION,
        }
