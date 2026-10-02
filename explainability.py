"""
Post-hoc SHAP-based explainability module (Section 3.8 and 4.9 of the
manuscript). The explainer operates *after* training and never touches the
model's learned parameters or predictions.

Because RA-FogNet consumes a length-T sequence per modality, we summarize
each window by its per-modality mean (a compact, tractable representation
for SHAP's KernelExplainer) and wrap the trained model in a predict_fn that
broadcasts a modality-level perturbation back across the full window before
calling the model. This gives modality-level Shapley attributions
(phi_R, phi_F, phi_G, phi_H, phi_V, phi_L; Eq. 43) consistent with the
`E_x` explanation vector described in the manuscript.
"""
from typing import Dict, List

import numpy as np
import torch

from .data import MODALITY_NAMES, MODALITY_DIMS

try:
    import shap
except ImportError:  # pragma: no cover
    shap = None


class ModalityLevelExplainer:
    """Wraps a trained RA-FogNet model for modality-level SHAP attribution."""

    def __init__(self, model, background_batch: Dict[str, torch.Tensor], device, target_class: int = None):
        if shap is None:
            raise ImportError("Please `pip install shap` to use ModalityLevelExplainer.")
        self.model = model.eval()
        self.device = device
        self.target_class = target_class
        self.modality_names = MODALITY_NAMES
        self.modality_dims = MODALITY_DIMS
        # Background: mean window per modality per sample -> (N, M) "modality present" mask baseline
        self.background_windows = {m: v.to(device) for m, v in background_batch.items()}
        self.n_background = next(iter(self.background_windows.values())).shape[0]

    def _predict(self, mask_matrix: np.ndarray) -> np.ndarray:
        """
        mask_matrix: (n_samples, M) in [0, 1]; 1 = keep original modality window,
        0 = replace with the background (mean) window for that modality.
        Returns softmax probabilities (n_samples, n_classes) for `target_class`
        or the full distribution if target_class is None.
        """
        n_samples = mask_matrix.shape[0]
        probs = []
        batch_size = 32
        for start in range(0, n_samples, batch_size):
            chunk = mask_matrix[start:start + batch_size]
            batch_modalities = {}
            for i, m in enumerate(self.modality_names):
                orig = self.background_windows[m][:chunk.shape[0]]  # (b, T, dim)
                mean_val = orig.mean(dim=1, keepdim=True)  # (b, 1, dim) baseline
                mask = torch.tensor(chunk[:, i], dtype=torch.float32, device=self.device).view(-1, 1, 1)
                batch_modalities[m] = mask * orig + (1 - mask) * mean_val
            with torch.no_grad():
                logits, _ = self.model(batch_modalities)
                p = torch.softmax(logits, dim=-1).cpu().numpy()
            probs.append(p)
        probs = np.concatenate(probs, axis=0)
        return probs[:, self.target_class] if self.target_class is not None else probs

    def explain(self, n_samples_bg: int = 20, target_class: int = None) -> np.ndarray:
        """
        Returns Shapley values of shape (M,) for the requested class,
        computed with shap.KernelExplainer over the binary modality-mask
        space (Eq. 42).
        """
        self.target_class = target_class if target_class is not None else self.target_class
        M = len(self.modality_names)
        background = np.zeros((1, M))  # all modalities replaced by their mean (baseline)
        explainer = shap.KernelExplainer(self._predict, background)
        instance = np.ones((1, M))  # all modalities present (actual observation)
        shap_values = explainer.shap_values(instance, nsamples=min(2 ** M, 256))
        return np.array(shap_values).reshape(-1)


def global_modality_importance(model, dataset, device, n_instances: int = 50, target_class: int = None) -> Dict[str, float]:
    """
    Aggregates modality-level |phi_i| over `n_instances` test windows to
    produce the global feature-importance ranking shown in Fig. 2.
    """
    from .data import collate_fn
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    totals = {m: 0.0 for m in MODALITY_NAMES}
    count = 0
    for modality_batch, label, _ in loader:
        if count >= n_instances:
            break
        explainer = ModalityLevelExplainer(model, modality_batch, device)
        phi = explainer.explain(target_class=target_class if target_class is not None else int(label.item()))
        for m, v in zip(MODALITY_NAMES, phi):
            totals[m] += abs(v)
        count += 1
    total_sum = sum(totals.values()) + 1e-12
    return {m: (v / total_sum) for m, v in totals.items()}  # Eq. 44 normalized importance


def local_explanation(model, modality_batch, label: int, device) -> Dict[str, float]:
    """Local SHAP explanation (E_x, Eq. 43) for a single instance -> Fig. 3."""
    explainer = ModalityLevelExplainer(model, modality_batch, device)
    phi = explainer.explain(target_class=label)
    return dict(zip(MODALITY_NAMES, phi.tolist()))
