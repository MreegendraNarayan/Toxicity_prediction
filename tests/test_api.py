from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

import api
from toxicity.inference.constants import TOX21_TASKS
from toxicity.inference.models import build_model
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


@pytest.fixture()
def client(tmp_path, monkeypatch):
    write_checkpoint(tmp_path / "Enhanced-GIN_best.pt", "Enhanced-GIN", "gin")
    write_checkpoint(tmp_path / "Enhanced-GCN_best.pt", "Enhanced-GCN", "gcn")
    monkeypatch.setattr(api, "predictor_service", ToxicityPredictor(checkpoint_dir=str(tmp_path)))
    return TestClient(api.app)


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_models_report_compatible_checkpoints(client):
    response = client.get("/models")
    data = response.json()

    assert response.status_code == 200
    assert data["default_model"] == "Enhanced-GIN"
    assert all(model["compatible"] for model in data["models"])


def test_predict_accepts_query_params(client):
    response = client.post("/predict", params={"smiles": ASPIRIN, "model": "Enhanced-GIN"})
    data = response.json()

    assert response.status_code == 200
    assert data["success"] is True
    assert list(data["predictions"].keys()) == TOX21_TASKS


def test_predict_accepts_json_body(client):
    response = client.post("/predict", json={"smiles": ASPIRIN, "model": "Enhanced-GCN"})
    data = response.json()

    assert response.status_code == 200
    assert data["model"] == "Enhanced-GCN"
    assert list(data["predictions"].keys()) == TOX21_TASKS


def test_invalid_smiles_returns_400(client):
    response = client.post("/predict", json={"smiles": "not-a-smiles", "model": "Enhanced-GIN"})

    assert response.status_code == 400


def test_batch_accepts_sample_csv(client):
    csv_content = "smiles,molecule_name\nCC(=O)Oc1ccccc1C(=O)O,Aspirin\n"
    response = client.post(
        "/batch",
        params={"model": "Enhanced-GIN"},
        files={"file": ("sample_molecules.csv", csv_content, "text/csv")},
    )
    data = response.json()

    assert response.status_code == 200
    assert data["processed"] == 1
    assert data["errors"] == 0
    assert list(data["rows"][0].keys())[:2] == ["smiles", "molecule_name"]
    for task in TOX21_TASKS:
        assert task in data["rows"][0]


def test_batch_missing_smiles_column_returns_400(client):
    response = client.post(
        "/batch",
        files={"file": ("bad.csv", "name\nAspirin\n", "text/csv")},
    )

    assert response.status_code == 400
