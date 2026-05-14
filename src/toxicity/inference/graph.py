"""RDKit and PyTorch Geometric graph conversion for molecular inference."""

from typing import Iterable, Optional

import networkx as nx
import torch
from rdkit import Chem
from torch_geometric.data import Batch, Data


class InvalidSmilesError(ValueError):
    """Raised when a SMILES string cannot be parsed into a molecule."""


def validate_smiles(smiles: str) -> str:
    """Return a stripped SMILES string or raise for empty input."""
    if smiles is None:
        raise InvalidSmilesError("SMILES cannot be empty")

    cleaned = str(smiles).strip()
    if not cleaned:
        raise InvalidSmilesError("SMILES cannot be empty")

    return cleaned


def smiles_to_graph(smiles: str, labels: Optional[Iterable[float]] = None) -> Data:
    """Convert a SMILES string into a PyTorch Geometric graph.

    Node features are:
    1. atomic number
    2. normalized atom degree
    3. clustering coefficient
    4. betweenness centrality
    """
    cleaned = validate_smiles(smiles)
    mol = Chem.MolFromSmiles(cleaned)
    if mol is None:
        raise InvalidSmilesError(f"Invalid SMILES: {cleaned}")

    mol = Chem.AddHs(mol)
    num_nodes = mol.GetNumAtoms()
    if num_nodes == 0:
        raise InvalidSmilesError(f"Invalid SMILES: {cleaned}")

    undirected_edges = []
    directed_edges = []
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        undirected_edges.append((begin, end))
        directed_edges.extend([(begin, end), (end, begin)])

    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    graph.add_edges_from(undirected_edges)

    atomic_numbers = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=torch.float32,
    ).unsqueeze(1)

    degrees = torch.tensor(
        [float(graph.degree(node)) for node in range(num_nodes)],
        dtype=torch.float32,
    ).unsqueeze(1)
    max_degree = torch.clamp(degrees.max(), min=1.0)
    degree_features = degrees / max_degree

    clustering = nx.clustering(graph)
    clustering_features = torch.tensor(
        [float(clustering.get(node, 0.0)) for node in range(num_nodes)],
        dtype=torch.float32,
    ).unsqueeze(1)

    betweenness = nx.betweenness_centrality(graph, normalized=True)
    betweenness_features = torch.tensor(
        [float(betweenness.get(node, 0.0)) for node in range(num_nodes)],
        dtype=torch.float32,
    ).unsqueeze(1)

    data = Data(
        x=torch.cat(
            [atomic_numbers, degree_features, clustering_features, betweenness_features],
            dim=1,
        ),
        edge_index=edge_index,
        smiles=cleaned,
    )

    if labels is not None:
        data.y = torch.tensor(list(labels), dtype=torch.float32)

    return data


def smiles_batch_to_pyg(smiles_values: Iterable[str]) -> Batch:
    """Convert an iterable of SMILES strings into a PyG batch."""
    return Batch.from_data_list([smiles_to_graph(smiles) for smiles in smiles_values])
