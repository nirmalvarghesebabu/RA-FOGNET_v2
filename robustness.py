"""
Robustness evaluation under controlled sensor degradation (Section 4.7,
Table 12 of the manuscript): Gaussian noise injection at sigma in
{0.10, 0.20, 0.30} on Z-score-normalized features (Eq. `eq:noise_levels`),
and a missing-sensor condition where one modality is entirely masked.
"""
from typing import Dict, List

import torch

from .data import MODALITY_NAMES
from .train import evaluate_dataset


def make_gaussian_noise_fn(sigma: float, generator: torch.Generator = None):
    """Returns a callable that adds N(0, sigma^2) noise to every modality tensor."""

    def _fn(modality_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for m, x in modality_batch.items():
            noise = torch.randn(x.shape, generator=generator) * sigma
            out[m] = x + noise
        return out

    return _fn


def make_missing_sensor_fn(modality: str):
    """Returns a callable that zero-masks the specified modality (missing-data representation)."""

    def _fn(modality_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(modality_batch)
        out[modality] = torch.zeros_like(out[modality])
        return out

    return _fn


def run_robustness_suite(
    model, test_ds, model_kind: str = "ra_fognet",
    noise_levels: List[float] = (0.10, 0.20, 0.30),
    missing_modalities: List[str] = None,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Runs the full robustness protocol and returns a dict keyed by condition
    name ("clean", "noise_10", "noise_20", "noise_30", "missing_<modality>")
    -> metrics dict, matching the rows/columns of Table 12.
    """
    gen = torch.Generator().manual_seed(seed)
    results = {"clean": evaluate_dataset(model, test_ds, model_kind)}

    for sigma in noise_levels:
        key = f"noise_{int(round(sigma * 100))}"
        results[key] = evaluate_dataset(
            model, test_ds, model_kind, noise_fn=make_gaussian_noise_fn(sigma, gen)
        )

    missing_modalities = missing_modalities or MODALITY_NAMES
    for modality in missing_modalities:
        key = f"missing_{modality}"
        results[key] = evaluate_dataset(
            model, test_ds, model_kind, noise_fn=make_missing_sensor_fn(modality)
        )

    # Aggregate "missing sensor" condition as the worst-case single-modality dropout,
    # matching the single "Missing Sensor" row reported in Table 12.
    worst = min((results[f"missing_{m}"]["accuracy"] for m in missing_modalities))
    results["missing_sensor_worst_case"] = {"accuracy": worst}
    return results
