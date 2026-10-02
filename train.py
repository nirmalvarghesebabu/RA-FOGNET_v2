"""
Training / evaluation loops for RA-FogNet and baseline models
(Section 4.2, Table 2 of the manuscript: Adam, lr=1e-4, batch=64,
max 100 epochs, patience=15, weight decay=1e-5, dropout=0.2).
"""
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import flatten_modalities, MODALITY_NAMES
from .losses import RAFogNetLoss
from .utils import EarlyStopping, classification_metrics, regression_metrics, get_device


DEFAULT_TRAIN_CONFIG = dict(
    lr=1e-4,
    weight_decay=1e-5,
    batch_size=64,
    max_epochs=100,
    patience=15,
)


def _forward_model(model, modality_batch, model_kind: str, device, return_aux=False):
    """Dispatch forward pass: RA-FogNet consumes a dict of modalities; baselines
    consume a single flattened tensor (B, T, D)."""
    modality_batch = {m: v.to(device) for m, v in modality_batch.items()}
    if model_kind == "ra_fognet":
        return model(modality_batch, return_aux=return_aux)
    else:
        x = flatten_modalities(modality_batch)
        out = model(x)
        return (*out, {}) if return_aux else out


def train_model(
    model: torch.nn.Module,
    train_ds,
    val_ds,
    model_kind: str = "ra_fognet",
    config: Optional[dict] = None,
    verbose: bool = True,
) -> Dict:
    """Full training loop with early stopping on validation accuracy."""
    cfg = {**DEFAULT_TRAIN_CONFIG, **(config or {})}
    device = get_device()
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               collate_fn=_collate(train_ds))
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             collate_fn=_collate(val_ds))

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = RAFogNetLoss(lambda_pred=cfg.get("lambda_pred", 1.0), lambda_att=cfg.get("lambda_att", 0.0))
    stopper = EarlyStopping(patience=cfg["patience"], mode="max")

    best_state = None
    history = []

    for epoch in range(cfg["max_epochs"]):
        model.train()
        t0 = time.time()
        for modality_batch, labels, vis in train_loader:
            labels, vis = labels.to(device), vis.to(device)
            optimizer.zero_grad()
            logits, vis_pred, aux = _forward_model(model, modality_batch, model_kind, device, return_aux=True)
            alpha = aux.get("alpha") if isinstance(aux, dict) else None
            loss, _ = criterion(logits, labels, vis_pred, vis, alpha)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        val_metrics = evaluate_model(model, val_loader, model_kind, device)
        history.append(val_metrics)
        improved = stopper.step(val_metrics["accuracy"])
        if improved:
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"[epoch {epoch+1:03d}] val_acc={val_metrics['accuracy']:.2f} "
                  f"val_f1={val_metrics['f1']:.2f} time={time.time()-t0:.1f}s")
        if stopper.should_stop:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"history": history, "model": model}


def _collate(dataset):
    from .data import collate_fn
    return collate_fn


@torch.no_grad()
def evaluate_model(model, loader, model_kind: str, device=None, noise_fn=None) -> Dict:
    """Evaluate accuracy/precision/recall/F1 and visibility regression metrics.

    `noise_fn(modality_batch) -> modality_batch` can be supplied to inject
    controlled degradation for the robustness experiments (Section 4.7).
    """
    device = device or get_device()
    model.eval()
    all_logits, all_labels, all_vis_pred, all_vis_true = [], [], [], []
    for modality_batch, labels, vis in loader:
        if noise_fn is not None:
            modality_batch = noise_fn(modality_batch)
        logits, vis_pred = _forward_model(model, modality_batch, model_kind, device, return_aux=False)
        all_logits.append(logits.cpu())
        all_labels.append(labels)
        if vis_pred is not None:
            all_vis_pred.append(vis_pred.cpu())
            all_vis_true.append(vis)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = logits.argmax(dim=-1).numpy()
    y_true = labels.numpy()
    metrics = classification_metrics(y_true, preds)

    if all_vis_pred:
        vis_pred = torch.cat(all_vis_pred, dim=0).numpy()
        vis_true = torch.cat(all_vis_true, dim=0).numpy()
        metrics.update({f"vis_{k}": v for k, v in regression_metrics(vis_true, vis_pred).items()})
    return metrics


def evaluate_dataset(model, dataset, model_kind: str, batch_size: int = 64, noise_fn=None) -> Dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=_collate(dataset))
    return evaluate_model(model, loader, model_kind, noise_fn=noise_fn)
