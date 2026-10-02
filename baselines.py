"""
Baseline architectures reproducing Table 6 of the manuscript. Each baseline
consumes the same concatenated multimodal input tensor of shape (B, T, D)
(D = sum of modality dims) and outputs classification logits plus an
optional visibility regression head, so all models share the identical
data pipeline, losses, and evaluation protocol as RA-FogNet.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import SinusoidalPositionalEncoding


class _Heads(nn.Module):
    """Shared classification + regression head used by every baseline."""

    def __init__(self, in_dim: int, n_classes: int = 4, predict_visibility: bool = True):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(0.2), nn.Linear(in_dim, n_classes)
        )
        self.predict_visibility = predict_visibility
        if predict_visibility:
            self.regressor = nn.Sequential(nn.Linear(in_dim, in_dim // 2), nn.GELU(), nn.Linear(in_dim // 2, 1))

    def forward(self, pooled):
        logits = self.classifier(pooled)
        vis = self.regressor(pooled).squeeze(-1) if self.predict_visibility else None
        return logits, vis


class CNNBaseline(nn.Module):
    """1D convolutional feature extraction + global pooling (Table 6: CNN)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_classes: int = 4, predict_visibility=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.heads = _Heads(hidden_dim, n_classes, predict_visibility)

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        h = self.conv(x.transpose(1, 2)).squeeze(-1)
        return self.heads(h)


class CNNLSTM(nn.Module):
    """CNN feature extractor + LSTM temporal encoder (Table 6: CNN-LSTM)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_classes: int = 4, predict_visibility=True):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(in_dim, hidden_dim, 3, padding=1), nn.GELU())
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.heads = _Heads(hidden_dim, n_classes, predict_visibility)

    def forward(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, T, H)
        out, _ = self.lstm(h)
        return self.heads(out.mean(dim=1))


class GRUModel(nn.Module):
    """Recurrent GRU temporal encoder (Table 6: GRU)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_classes: int = 4, predict_visibility=True):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.heads = _Heads(hidden_dim, n_classes, predict_visibility)

    def forward(self, x):
        out, _ = self.gru(F.gelu(self.proj(x)))
        return self.heads(out.mean(dim=1))


class BiLSTM(nn.Module):
    """Bidirectional LSTM temporal encoder (Table 6: BiLSTM)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_classes: int = 4, predict_visibility=True):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.heads = _Heads(hidden_dim * 2, n_classes, predict_visibility)

    def forward(self, x):
        out, _ = self.lstm(F.gelu(self.proj(x)))
        return self.heads(out.mean(dim=1))


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.pad = pad

    def forward(self, x):
        h = self.conv(x)
        if self.pad > 0:
            h = h[:, :, :-self.pad]
        return self.drop(self.act(h))


class TCN(nn.Module):
    """Dilated temporal convolution network (Table 6: TCN)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_classes: int = 4, predict_visibility=True):
        super().__init__()
        self.blocks = nn.Sequential(
            TCNBlock(in_dim, hidden_dim, dilation=1),
            TCNBlock(hidden_dim, hidden_dim, dilation=2),
            TCNBlock(hidden_dim, hidden_dim, dilation=4),
        )
        self.heads = _Heads(hidden_dim, n_classes, predict_visibility)

    def forward(self, x):
        h = self.blocks(x.transpose(1, 2)).transpose(1, 2)
        return self.heads(h.mean(dim=1))


class TransformerEncoderBaseline(nn.Module):
    """Plain multi-head self-attention encoder (Table 6: Transformer Encoder / Informer proxy)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64, n_heads: int = 4, n_layers: int = 2,
                 n_classes: int = 4, predict_visibility=True, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim)
        self.pos = SinusoidalPositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, hidden_dim * 4, dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.heads = _Heads(hidden_dim, n_classes, predict_visibility)

    def forward(self, x):
        h = self.pos(self.proj(x))
        h = self.encoder(h)
        return self.heads(h.mean(dim=1))


class GraphConvBaseline(nn.Module):
    """
    Simplified GCN baseline (Table 6: GCN). Builds a *fixed* (learned but
    time-invariant) adjacency over feature-chunks rather than the dynamic,
    reliability-weighted graph used by RA-FogNet.
    """

    def __init__(self, in_dim: int, n_nodes: int = 6, hidden_dim: int = 64,
                 n_classes: int = 4, predict_visibility=True):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = math.ceil(in_dim / n_nodes)
        pad_dim = self.node_dim * n_nodes
        self.pad = pad_dim - in_dim
        self.node_proj = nn.Linear(self.node_dim, hidden_dim)
        self.adj = nn.Parameter(torch.eye(n_nodes) + 0.1 * torch.randn(n_nodes, n_nodes))
        self.gc = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.heads = _Heads(hidden_dim, n_classes, predict_visibility)

    def forward(self, x):
        B, T, D = x.shape
        if self.pad > 0:
            x = F.pad(x, (0, self.pad))
        x = x.view(B, T, self.n_nodes, self.node_dim)
        h = F.gelu(self.node_proj(x))  # (B,T,N,H)
        A = F.softmax(self.adj, dim=-1)
        h = torch.einsum("mn,btnh->btmh", A, h)
        h = F.gelu(self.gc(h))
        h = h.mean(dim=2)  # aggregate nodes -> (B, T, H)
        return self.heads(h.mean(dim=1))


class GraphAttentionBaseline(nn.Module):
    """Simplified GAT baseline (Table 6: GAT) with learned attention over fixed nodes."""

    def __init__(self, in_dim: int, n_nodes: int = 6, hidden_dim: int = 64,
                 n_classes: int = 4, predict_visibility=True):
        super().__init__()
        self.n_nodes = n_nodes
        self.node_dim = math.ceil(in_dim / n_nodes)
        pad_dim = self.node_dim * n_nodes
        self.pad = pad_dim - in_dim
        self.node_proj = nn.Linear(self.node_dim, hidden_dim)
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.heads = _Heads(hidden_dim, n_classes, predict_visibility)

    def forward(self, x):
        B, T, D = x.shape
        if self.pad > 0:
            x = F.pad(x, (0, self.pad))
        x = x.view(B, T, self.n_nodes, self.node_dim)
        h = F.gelu(self.node_proj(x))  # (B,T,N,H)
        q, k = self.attn_q(h), self.attn_k(h)
        scores = torch.einsum("btmh,btnh->btmn", q, k) / math.sqrt(h.shape[-1])
        attn = F.softmax(scores, dim=-1)
        h = torch.einsum("btmn,btnh->btmh", attn, h)
        h = h.mean(dim=2)
        return self.heads(h.mean(dim=1))


class GraphTransformerBaseline(nn.Module):
    """Graph representation learning followed by a Transformer encoder (Table 6: Graph Transformer)."""

    def __init__(self, in_dim: int, n_nodes: int = 6, hidden_dim: int = 64,
                 n_classes: int = 4, predict_visibility=True, dropout=0.2):
        super().__init__()
        self.gat = GraphAttentionBaseline(in_dim, n_nodes, hidden_dim, n_classes, predict_visibility=False)
        # Re-use GAT's node encoder but keep the temporal sequence (skip GAT's own pooled heads)
        self.node_dim = self.gat.node_dim
        self.n_nodes = n_nodes
        self.pad = self.gat.pad
        self.pos = SinusoidalPositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 4, dropout, batch_first=True, activation="gelu")
        self.temporal = nn.TransformerEncoder(layer, 2)
        self.heads = _Heads(hidden_dim, n_classes, predict_visibility)

    def forward(self, x):
        B, T, D = x.shape
        xp = F.pad(x, (0, self.pad)) if self.pad > 0 else x
        xp = xp.view(B, T, self.n_nodes, self.node_dim)
        h = F.gelu(self.gat.node_proj(xp))
        q, k = self.gat.attn_q(h), self.gat.attn_k(h)
        scores = torch.einsum("btmh,btnh->btmn", q, k) / math.sqrt(h.shape[-1])
        attn = F.softmax(scores, dim=-1)
        h = torch.einsum("btmn,btnh->btmh", attn, h).mean(dim=2)  # (B,T,H)
        h = self.pos(h)
        h = self.temporal(h)
        return self.heads(h.mean(dim=1))


class AttentionFusionNetwork(nn.Module):
    """
    Conventional attention-based multimodal fusion *without* reliability
    estimation (Table 6: Attention Fusion Network) — i.e. RA-FogNet's
    fusion stage with the reliability bias term removed and no graph /
    temporal-Transformer stages, matching the isolated fusion-only baseline
    used for comparison in Table 7.
    """

    def __init__(self, modality_dims: dict, hidden_dim: int = 64, n_classes: int = 4, predict_visibility=True):
        super().__init__()
        from .modules import MultiModalEncoder, ReliabilityAttentionFusion
        self.encoder = MultiModalEncoder(modality_dims, hidden_dim)
        self.fusion = ReliabilityAttentionFusion(hidden_dim, lam=1.0)
        self.temporal_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.heads = _Heads(hidden_dim, n_classes, predict_visibility)

    def forward(self, modality_batch, return_aux: bool = False):
        Z = self.encoder(modality_batch)  # (B,T,M,H)
        rho = torch.zeros(Z.shape[0], Z.shape[1], Z.shape[2], device=Z.device)
        Z_F, alpha = self.fusion(Z, rho, use_reliability=False)
        out, _ = self.temporal_gru(Z_F)
        logits, vis = self.heads(out.mean(dim=1))
        if return_aux:
            return logits, vis, {"alpha": alpha}
        return logits, vis


BASELINE_REGISTRY = {
    "cnn": CNNBaseline,
    "cnn_lstm": CNNLSTM,
    "gru": GRUModel,
    "bilstm": BiLSTM,
    "tcn": TCN,
    "transformer": TransformerEncoderBaseline,
    "informer": TransformerEncoderBaseline,  # proxy: efficient-attention variant not required for parity checks
    "gcn": GraphConvBaseline,
    "gat": GraphAttentionBaseline,
    "graph_transformer": GraphTransformerBaseline,
}
