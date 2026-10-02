"""
Core building blocks of RA-FogNet, implementing Sections 3.3-3.7 of the
manuscript:
    - Modality-specific encoders                 (Eq. 5-7)
    - Local uncertainty / reliability estimation  (Eq. 8-12)
    - Reliability-aware attention fusion          (Eq. 13-18)
    - Sensor-interaction graph learning           (Eq. 19-24)
    - Sinusoidal positional encoding              (Eq. 26-28)
"""
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityEncoder(nn.Module):
    """Eq. 5-6: Z_{m,t} = sigma(W_m X_{m,t} + b_m)."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, in_dim) -> (B, T, hidden_dim)
        return self.net(x)


class MultiModalEncoder(nn.Module):
    """Applies a dedicated encoder per modality (Eq. 7)."""

    def __init__(self, modality_dims: Dict[str, int], hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.encoders = nn.ModuleDict({
            m: ModalityEncoder(d, hidden_dim, dropout) for m, d in modality_dims.items()
        })

    def forward(self, modality_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Returns Z_t of shape (B, T, M, H)."""
        encoded = [self.encoders[m](modality_batch[m]) for m in self.modality_names]
        return torch.stack(encoded, dim=2)  # (B, T, M, H)


class ReliabilityEstimator(nn.Module):
    """
    Eq. 8-12: local temporal uncertainty and reliability coefficient.

    For each modality, the local window mean and dispersion are computed
    with a causal moving window of length `window`, and converted into a
    reliability coefficient rho = 1 / (1 + U).
    """

    def __init__(self, window: int = 5):
        super().__init__()
        self.window = window

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Z: (B, T, M, H) modality representations.
        Returns rho: (B, T, M) reliability coefficients in (0, 1].
        """
        B, T, M, H = Z.shape
        w = self.window
        # Causal moving average via unfold-free cumulative trick (padded)
        pad = Z.new_zeros(B, w - 1, M, H)
        Z_padded = torch.cat([pad, Z], dim=1)  # (B, T+w-1, M, H)
        windows = Z_padded.unfold(1, w, 1)  # (B, T, M, H, w) -> last dim is time
        windows = windows.permute(0, 1, 2, 4, 3)  # (B, T, M, w, H)
        mean = windows.mean(dim=3, keepdim=True)  # (B, T, M, 1, H)
        U = ((windows - mean) ** 2).sum(dim=-1).mean(dim=3)  # (B, T, M)
        rho = 1.0 / (1.0 + U)
        return rho


class ReliabilityAttentionFusion(nn.Module):
    """
    Eq. 13-18: reliability-aware attention fusion across modalities.

    e_{m,t} = (q_t^T k_{m,t}) / sqrt(d) + lambda * rho_{m,t}
    alpha   = softmax_m(e)
    Z_F     = sum_m alpha_m * Z_m
    """

    def __init__(self, hidden_dim: int, lam: float = 1.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lam = lam
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_seed = nn.Parameter(torch.randn(hidden_dim) * 0.02)

    def forward(self, Z: torch.Tensor, rho: torch.Tensor, use_reliability: bool = True):
        """
        Z:   (B, T, M, H)
        rho: (B, T, M)
        Returns:
            Z_F:   (B, T, H)    fused representation
            alpha: (B, T, M)    modality attention weights
        """
        B, T, M, H = Z.shape
        # Query is derived from the mean modality representation (context) plus a learned seed
        context = Z.mean(dim=2)  # (B, T, H)
        q = self.query_proj(context + self.query_seed)  # (B, T, H)
        k = self.key_proj(Z)  # (B, T, M, H)

        scores = torch.einsum("bth,btmh->btm", q, k) / math.sqrt(H)
        if use_reliability:
            scores = scores + self.lam * rho
        alpha = F.softmax(scores, dim=-1)  # (B, T, M)
        Z_F = torch.einsum("btm,btmh->bth", alpha, Z)
        return Z_F, alpha


class SensorGraphLayer(nn.Module):
    """
    Eq. 19-24: builds a per-timestep sensor-interaction graph from
    reliability-weighted modality embeddings and applies normalized graph
    convolution (single or stacked layers).
    """

    def __init__(self, hidden_dim: int, n_layers: int = 2, eps: float = 1e-8, dropout: float = 0.1):
        super().__init__()
        self.eps = eps
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def _build_adjacency(self, H: torch.Tensor) -> torch.Tensor:
        """H: (B, T, M, D) node features -> normalized adjacency (B, T, M, M)."""
        H_norm = F.normalize(H, p=2, dim=-1, eps=self.eps)
        A = torch.einsum("btmd,btnd->btmn", H_norm, H_norm)
        A = torch.clamp(A, min=0.0)  # Eq. 21: ReLU-clipped cosine similarity
        M = A.shape[-1]
        I = torch.eye(M, device=A.device, dtype=A.dtype)
        A_tilde = A + I  # Eq. 22: self-connections
        deg = A_tilde.sum(dim=-1)  # (B, T, M)
        deg_inv_sqrt = torch.pow(deg + self.eps, -0.5)
        D_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
        A_norm = D_inv_sqrt @ A_tilde @ D_inv_sqrt  # Eq. 24 normalization term
        return A_norm

    def forward(self, h_weighted: torch.Tensor) -> torch.Tensor:
        """
        h_weighted: (B, T, M, D) reliability-weighted node features (Eq. 20).
        Returns graph-enhanced node features (B, T, M, D).
        """
        H = h_weighted
        for layer in self.layers:
            A_norm = self._build_adjacency(H)
            H = torch.einsum("btmn,btnd->btmd", A_norm, H)
            H = self.dropout(F.gelu(layer(H)))
        return self.norm(H + h_weighted)


class SinusoidalPositionalEncoding(nn.Module):
    """Eq. 27-28: standard sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim)
        return x + self.pe[:, : x.size(1), :]


class TemporalTransformerEncoder(nn.Module):
    """Eq. 29-35: multi-head self-attention temporal encoder + mean pooling."""

    def __init__(self, dim: int, n_heads: int = 4, n_layers: int = 2,
                 ff_dim: int = 256, dropout: float = 0.2, max_len: int = 512):
        super().__init__()
        self.pos_enc = SinusoidalPositionalEncoding(dim, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=n_heads, dim_feedforward=ff_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, Z_T: torch.Tensor) -> torch.Tensor:
        """Z_T: (B, T, dim). Returns pooled representation (B, dim)."""
        E = self.pos_enc(Z_T)
        H = self.encoder(E)  # (B, T, dim)
        z = H.mean(dim=1)  # Eq. 35: temporal aggregation
        return z
