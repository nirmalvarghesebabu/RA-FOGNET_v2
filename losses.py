"""Training objective: classification + visibility forecasting + attention entropy reg (Eq. 52-56)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention_entropy_regularizer(alpha: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Eq. 54: L_att = -sum_m alpha_m log(alpha_m + eps), averaged over batch/time."""
    ent = -(alpha * torch.log(alpha + eps)).sum(dim=-1)
    return ent.mean()


class RAFogNetLoss(nn.Module):
    """Eq. 55: L_total = L_cls + lambda_pred * L_pred + lambda_att * L_att."""

    def __init__(self, lambda_pred: float = 1.0, lambda_att: float = 0.0):
        super().__init__()
        self.lambda_pred = lambda_pred
        self.lambda_att = lambda_att
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()

    def forward(self, logits, labels, vis_pred=None, vis_true=None, alpha=None):
        loss = self.ce(logits, labels)
        parts = {"cls": loss.item()}
        if vis_pred is not None and vis_true is not None:
            l_pred = self.mse(vis_pred, vis_true)
            loss = loss + self.lambda_pred * l_pred
            parts["pred"] = l_pred.item()
        if self.lambda_att > 0 and alpha is not None:
            l_att = attention_entropy_regularizer(alpha)
            loss = loss + self.lambda_att * l_att
            parts["att"] = l_att.item()
        parts["total"] = loss.item()
        return loss, parts
