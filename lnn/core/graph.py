from __future__ import annotations

import torch
import torch.nn as nn

from lnn.core.cfc import CfCNetwork
from lnn.core.ltc import LTCNetwork


class GraphSnapshotEncoder(nn.Module):
    """Small dependency-free graph encoder for dynamic graph snapshots."""

    def __init__(self, node_feature_size: int, hidden_size: int = 32, output_size: int = 32) -> None:
        super().__init__()
        self.self_proj = nn.Linear(node_feature_size, hidden_size)
        self.neighbor_proj = nn.Linear(node_feature_size, hidden_size)
        self.output = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, output_size),
            nn.SiLU(),
        )

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if node_features.dim() != 4:
            raise ValueError("node_features must have shape [batch, time, nodes, features]")
        if adjacency.dim() == 3:
            adjacency = adjacency.unsqueeze(0).expand(node_features.shape[0], -1, -1, -1)
        if adjacency.dim() != 4:
            raise ValueError("adjacency must have shape [time, nodes, nodes] or [batch, time, nodes, nodes]")

        batch, seq_len, num_nodes, _ = node_features.shape
        if adjacency.shape[:2] != (batch, seq_len) or adjacency.shape[-2:] != (num_nodes, num_nodes):
            raise ValueError("adjacency shape must match batch, time, and node dimensions")

        eye = torch.eye(num_nodes, device=node_features.device, dtype=node_features.dtype)
        adjacency = adjacency.to(device=node_features.device, dtype=node_features.dtype)
        adjacency = adjacency + eye.view(1, 1, num_nodes, num_nodes)
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        normalized = adjacency / degree
        neighbor_features = torch.einsum("btij,btjf->btif", normalized, node_features)
        encoded_nodes = self.self_proj(node_features) + self.neighbor_proj(neighbor_features)
        encoded_nodes = self.output(encoded_nodes)
        return encoded_nodes.mean(dim=2)


class GraphLNNPredictor(nn.Module):
    """
    GNN encoder + temporal LNN for graph-level time-series prediction.

    The model accepts a dict with `node_features`, `adjacency`, and optional
    `dt`/`mask`. It returns a graph-level prediction from the final time step.
    """

    def __init__(
        self,
        node_feature_size: int,
        graph_feature_size: int = 32,
        hidden_size: int = 48,
        output_size: int = 1,
        recurrent_type: str = "cfc",
    ) -> None:
        super().__init__()
        recurrent_type = recurrent_type.lower()
        if recurrent_type not in {"cfc", "ltc", "gru"}:
            raise ValueError("recurrent_type must be cfc, ltc, or gru")
        self.recurrent_type = recurrent_type
        self.encoder = GraphSnapshotEncoder(
            node_feature_size,
            hidden_size=graph_feature_size,
            output_size=graph_feature_size,
        )
        if recurrent_type == "cfc":
            self.recurrent = CfCNetwork(graph_feature_size, hidden_size, output_size, return_sequences=False)
        elif recurrent_type == "ltc":
            self.recurrent = LTCNetwork(graph_feature_size, hidden_size, output_size, ode_method="euler")
        else:
            self.recurrent = nn.GRU(graph_feature_size, hidden_size, batch_first=True)
            self.readout = nn.Linear(hidden_size, output_size)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        graph_sequence = self.encoder(batch["node_features"], batch["adjacency"])
        if self.recurrent_type == "gru":
            output, _ = self.recurrent(graph_sequence)
            return self.readout(output[:, -1, :])
        output = self.recurrent(graph_sequence, dt=batch.get("dt"), mask=batch.get("mask"))
        if output.dim() == 3:
            output = output[:, -1, :]
        return output
