"""
Ablation study runner (Section 4.6, Table 9 of the manuscript). Reuses the
single RAFogNet implementation with its `use_reliability` / `use_graph` /
`use_transformer` flags toggled, so every variant is trained and evaluated
identically to the full model.
"""
from typing import Dict, List

from .model import RAFogNet
from .train import train_model, evaluate_dataset

ABLATION_VARIANTS = [
    # (name, use_reliability, use_graph, use_transformer)
    ("Basic Sensor Fusion", False, False, False),
    ("Transformer Fusion", False, False, True),
    ("Reliability Attention + Transformer", True, False, True),
    ("Reliability Attention + Graph Learning", True, True, False),
    ("Graph + Transformer", False, True, True),
    ("Complete RA-FogNet", True, True, True),
]


def run_ablation_suite(
    modality_dims: Dict[str, int],
    train_ds,
    val_ds,
    test_ds,
    model_config: dict = None,
    train_config: dict = None,
    verbose: bool = True,
) -> List[Dict]:
    """Trains every ablation variant and returns a list of result dicts."""
    model_config = model_config or {}
    results = []
    for name, use_rel, use_graph, use_transformer in ABLATION_VARIANTS:
        if verbose:
            print(f"\n=== Ablation variant: {name} ===")
        cfg = {**model_config, "use_reliability": use_rel, "use_graph": use_graph,
               "use_transformer": use_transformer}
        model = RAFogNet(modality_dims=modality_dims, **cfg)
        train_model(model, train_ds, val_ds, model_kind="ra_fognet",
                    config=train_config, verbose=verbose)
        metrics = evaluate_dataset(model, test_ds, model_kind="ra_fognet")
        results.append({"variant": name, **metrics})
    return results
