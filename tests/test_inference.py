from pathlib import Path

import torch
from torch_geometric.data import Batch

from toxicity.inference.constants import TOX21_TASKS
from toxicity.inference.graph import smiles_to_graph
from toxicity.inference.models import build_model, predict_probabilities
from toxicity.inference.predictor import ToxicityPredictor

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"


def write_checkpoint(path: Path, model_name: str, architecture: str) -> None:
    model = build_model(architecture, num_tasks=len(TOX21_TASKS))
    torch.save(
        {
            "model_name": model_name,
            "architecture": architecture,
            "task_names": TOX21_TASKS,
            "feature_version": "rdkit-topology-v1",
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def test_smiles_to_graph_has_four_node_features():
    graph = smiles_to_graph(ASPIRIN)

    assert graph.x.shape[0] > 0
    assert graph.x.shape[1] == 4
    assert graph.edge_index.shape[0] == 2


def test_gcn_and_gin_forward_shapes():
    graph = smiles_to_graph(ASPIRIN)
    batch = Batch.from_data_list([graph])

    for architecture in ("gcn", "gin"):
        model = build_model(architecture, num_tasks=len(TOX21_TASKS))
        model.eval()
        probabilities = predict_probabilities(model, batch)

        assert probabilities.shape == (1, len(TOX21_TASKS))
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)


def test_predictor_returns_task_keyed_probabilities(tmp_path):
    write_checkpoint(tmp_path / "Enhanced-GIN_best.pt", "Enhanced-GIN", "gin")
    predictor = ToxicityPredictor(checkpoint_dir=str(tmp_path))

    result = predictor.predict_one(ASPIRIN, "Enhanced-GIN")

    assert result["model"] == "Enhanced-GIN"
    assert result["smiles"] == ASPIRIN
    assert list(result["predictions"].keys()) == TOX21_TASKS
    assert 0 <= result["average_toxicity"] <= 1
    assert result["risk_label"] in {"low", "moderate", "high"}
