"""
RA-FogNet: full model assembly (Sections 3.1-3.9 of the manuscript).

The model exposes boolean flags `use_reliability`, `use_graph`, and
`use_transformer` so that a single implementation can reproduce every row
of the ablation study (Table 9): disabling `use_graph` and
`use_transformer` collapses the model to plain attention-weighted sensor
fusion; disabling `use_reliability` removes the reliability bias term from
the attention scores (Eq. 15), turning the fusion stage into a standard
multimodal attention baseline ("Attention Fusion Network").
"""
from typing import Dict, Optional

import torch
import torch.nn as nn

from .modules import (
    MultiModalEncoder,
    ReliabilityEstimator,
    ReliabilityAttentionFusion,
    SensorGraphLayer,
    TemporalTransformerEncoder,
)


class RAFogNet(nn.Module):
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 64,
        n_classes: int = 4,
        reliability_window: int = 5,
        reliability_lambda: float = 1.0,
        graph_layers: int = 2,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        transformer_ff: int = 256,
        dropout: float = 0.2,
        use_reliability: bool = True,
        use_graph: bool = True,
        use_transformer: bool = True,
        predict_visibility: bool = True,
    ):
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.M = len(self.modality_names)
        self.hidden_dim = hidden_dim
        self.use_reliability = use_reliability
        self.use_graph = use_graph
        self.use_transformer = use_transformer
        self.predict_visibility = predict_visibility

        self.encoder = MultiModalEncoder(modality_dims, hidden_dim, dropout)
        self.reliability_estimator = ReliabilityEstimator(window=reliability_window)
        self.fusion = ReliabilityAttentionFusion(hidden_dim, lam=reliability_lambda)

        if use_graph:
            self.graph = SensorGraphLayer(hidden_dim, n_layers=graph_layers, dropout=dropout)

        if use_transformer:
            self.temporal = TemporalTransformerEncoder(
                hidden_dim, n_heads=transformer_heads, n_layers=transformer_layers,
                ff_dim=transformer_ff, dropout=dropout,
            )
        else:
            # Non-Transformer ablation path: simple temporal mean + GRU fallback
            self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )
        if predict_visibility:
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(self, modality_batch: Dict[str, torch.Tensor], return_aux: bool = False):
        """
        modality_batch: dict of (B, T, dim_m) tensors, one per modality.
        Returns:
            logits: (B, n_classes)
            vis_pred: (B,) or None
            aux: dict with attention weights / graph embeddings if return_aux
        """
        Z = self.encoder(modality_batch)  # (B, T, M, H)
        rho = self.reliability_estimator(Z)  # (B, T, M)
        Z_F, alpha = self.fusion(Z, rho, use_reliability=self.use_reliability)  # (B,T,H),(B,T,M)

        if self.use_graph:
            # Eq. 20: reliability-weighted node features feed the graph module
            h_weighted = Z * alpha.unsqueeze(-1)  # (B, T, M, H)
            H_graph = self.graph(h_weighted)  # (B, T, M, H)
            Z_G = H_graph.mean(dim=2)  # aggregate graph nodes -> (B, T, H)
        else:
            Z_G = Z_F

        if self.use_transformer:
            z_pooled = self.temporal(Z_G)  # (B, H)
        else:
            out, _ = self.temporal_gru(Z_G)
            z_pooled = out.mean(dim=1)

        logits = self.classifier(z_pooled)
        vis_pred = self.regressor(z_pooled).squeeze(-1) if self.predict_visibility else None

        if return_aux:
            return logits, vis_pred, {"alpha": alpha, "rho": rho, "z_pooled": z_pooled}
        return logits, vis_pred


def build_model_from_config(cfg: dict, modality_dims: Dict[str, int]) -> RAFogNet:
    """Convenience factory used by training scripts and ablation runs."""
    return RAFogNet(
        modality_dims=modality_dims,
        hidden_dim=cfg.get("hidden_dim", 64),
        n_classes=cfg.get("n_classes", 4),
        reliability_window=cfg.get("reliability_window", 5),
        reliability_lambda=cfg.get("reliability_lambda", 1.0),
        graph_layers=cfg.get("graph_layers", 2),
        transformer_heads=cfg.get("transformer_heads", 4),
        transformer_layers=cfg.get("transformer_layers", 2),
        transformer_ff=cfg.get("transformer_ff", 256),
        dropout=cfg.get("dropout", 0.2),
        use_reliability=cfg.get("use_reliability", True),
        use_graph=cfg.get("use_graph", True),
        use_transformer=cfg.get("use_transformer", True),
        predict_visibility=cfg.get("predict_visibility", True),
    )
